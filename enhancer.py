import cv2
import numpy as np
import scipy
from rembg import remove
from PIL import Image

# ------------------ SEGMENTATION ------------------ #

def get_foreground_mask(image_rgb):
    result_rgba = remove(image_rgb)
    alpha = result_rgba[:, :, 3]
    return (alpha > 127).astype(np.uint8)  # Binary mask

def feather_mask(mask, feather_radius=15):
    blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather_radius)
    return np.clip(blurred_mask, 0, 1)[..., None]  # Shape: (H, W, 1)

def get_mask_for_rgb(mask):
    return mask[...,None] # Shape: (H,W,1)

# ------------------ BACKGROUND ------------------ #

def blur_background(img, mask, ksize=101):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    mask_rgb = get_mask_for_rgb(mask)
    return (mask_rgb * img + (1 - mask_rgb) * blurred).astype(np.uint8)

def white_background(image, mask):
    white = np.ones_like(image, dtype=np.uint8) * 255
    mask_rgb = get_mask_for_rgb(mask)
    return (mask_rgb * image + (1 - mask_rgb) * white).astype(np.uint8)

def rm_background_segmentation(img):
    fg_rgba = remove(Image.fromarray(img))
    fg_rgba = np.array(fg_rgba)
    return fg_rgba

def build_bounding_rect(img):
    H,W,C = img.shape
    init_mask = get_foreground_mask(img)
    ys, xs = np.nonzero(init_mask)
    ymin = max(np.min(ys) - 10, 0)
    ymax = max(np.max(ys) - 10, 0)
    xmin = max(np.min(xs) - 10, 0)
    xmax = max(np.max(xs) - 10, 0)
    w = min(xmax - xmin + 1 + 20, W - xmin - 1)
    h = min(ymax - ymin + 1 + 20, H - ymin - 1)
    rect = (xmin, ymin, w, h)
    return rect

def rm_background_grabcut(img):
    mask = np.ones(img.shape[:2], np.uint8)
    rect = build_bounding_rect(img)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    binary_mask = np.where((mask==2)|(mask==0), 0, 1).astype(np.uint8)
    result = img * binary_mask[:, :, np.newaxis]
    return binary_mask, result

def get_combo_mask(mask_rembg, mask_gc):
    return (mask_rembg & mask_gc).astype(int)

def get_masked_img(img, mask):
    return img * mask[:, :, np.newaxis]

# ------------------ GRADIENT DOMAIN ------------------ #

def compute_gradients(img):
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, :-1] = img[:, 1:] - img[:, :-1]
    dy[:-1, :] = img[1:, :] - img[:-1, :]
    return dx, dy

def enhance_highfreq_gradients(img, sigma=5, alpha=1.5):
    img_blur = cv2.GaussianBlur(img, (0, 0), sigma)
    high_frequency = img - img_blur
    dx, dy = compute_gradients(high_frequency)
    dx_enh = dx * alpha
    dy_enh = dy * alpha
    return dx_enh, dy_enh, img_blur

def get_gradient_diff(dx, dy):
    div = np.zeros_like(dx)
    div[:, :-1] += dx[:, :-1]
    div[:, 1:] -= dx[:, :-1]
    div[:-1, :] += dy[:-1, :]
    div[1:, :] -= dy[:-1, :]
    return div

def get_A(h, w):
    N = h * w
    A = scipy.sparse.lil_matrix((N, N), dtype=np.float32)
    im2var = np.arange(N).reshape(h, w)
    for y in range(h):
        for x in range(w):
            e = im2var[y,x]
            A[e,e] = -4
            if x == 0 or x == w-1: A[e,e] += 1
            if y == 0 or y == h-1: A[e,e] += 1
            if x > 0: A[e, im2var[y,x-1]] = 1
            if x < w - 1: A[e, im2var[y,x+1]] = 1
            if y > 0: A[e, im2var[y-1,x]] = 1
            if y < h - 1: A[e, im2var[y+1,x]] = 1
    return A.tocsr()

def poisson_solve(enh_grad_div, img_orig):
    h, w = enh_grad_div.shape
    A = get_A(h,w)
    b = enh_grad_div.flatten()
    b_boundary = img_orig.copy()
    b_boundary[1:-1, 1:-1] = 0
    dx, dy = compute_gradients(b_boundary)
    b += get_gradient_diff(dx, dy).flatten()
    x = scipy.sparse.linalg.lsqr(A, b)[0]
    return x.reshape((h, w))

def get_gradient_domain_contrast_enhancement(original_rgb_img, mask):
    blurred_output = blur_background(original_rgb_img, mask, ksize=51)
    img_rgb = original_rgb_img.astype(np.float32)
    final_result = np.zeros(img_rgb.shape)
    for c in range(3):
        dx_enh, dy_enh, _ = enhance_highfreq_gradients(img_rgb[:,:,c])
        enh_grad_diff = get_gradient_diff(dx_enh, dy_enh)
        reconstructed = poisson_solve(enh_grad_diff, img_rgb[:,:,c])
        final_result[:,:,c] = reconstructed + blurred_output[:,:,c]
    return np.clip(final_result, 0, 255).astype(np.uint8)

# ------------------ COLOR & LIGHT ------------------ #

def white_balance_and_tone_mapping(img_rgb):
    img = img_rgb / 255.0
    red_avg, green_avg, blue_avg = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    avg = (red_avg + green_avg + blue_avg) / 3.0
    color_cast = max([abs(red_avg - avg), abs(green_avg - avg), abs(blue_avg - avg)])
    if color_cast > 0.02:
        white_balance_strength = min(color_cast / avg, 0.4)
        img[:,:,0] *= 1.0 + white_balance_strength * ((avg - red_avg) / (red_avg + 1e-6))
        img[:,:,1] *= 1.0 + white_balance_strength * ((avg - green_avg) / (green_avg + 1e-6))
        img[:,:,2] *= 1.0 + white_balance_strength * ((avg - blue_avg) / (blue_avg + 1e-6))
        img = np.clip(img, 0, 1)
    img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel = img_lab[:,:,0].astype(np.float32) / 255.0
    low_percentile, high_percentile = np.percentile(l_channel, (1, 99))
    l_channel = np.clip((l_channel - low_percentile)/(high_percentile - low_percentile +1e-6), 0, 1)
    if np.mean(l_channel > 0.95) > 0.02:
        l_channel[l_channel > 0.95] = 0.95 + 0.5 * (l_channel[l_channel > 0.95] - 0.95)
    if np.mean(l_channel < 0.1) > 0.05:
        l_channel[l_channel < 0.1] = np.power(l_channel[l_channel < 0.1], 0.8)
    l_channel = np.power(l_channel, 0.95)
    l_channel = np.clip(0.5 + 1.1 * (l_channel - 0.5), 0, 1)
    img_lab[:,:,0] = (l_channel * 255).astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

def add_shadow(img, mask):
    shadow = mask.copy()
    h, w = shadow.shape[:2]
    src_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])
    dst_pts = np.float32([[0, h*0.3], [w*0.6, h*0.3], [0,h], [w, h*0.95]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    shadow_warped = cv2.warpPerspective(shadow.astype(np.float32), M.astype(np.float32), (w, h))
    shadow_blur = cv2.GaussianBlur(shadow_warped, (51, 51), 0)
    shadow_bgr = cv2.merge([shadow_blur]*3)
    shadow_final = (shadow_bgr * 0.4).astype(np.float32)
    background = np.ones_like(img)
    shadow = ((background - shadow_final) * 255.0).astype(np.uint8)
    return (get_mask_for_rgb(mask) * img) + (get_mask_for_rgb(1-mask) * shadow)

# ------------------ MAIN PIPELINE ------------------ #

def enhance_image_pipeline(image_rgb, step_callback=None):
    if step_callback: step_callback("Segmenting product from background")
    mask_rembg = get_foreground_mask(image_rgb)

    if step_callback: step_callback("Enhancing contrast using gradient domain")
    gdce_img = get_gradient_domain_contrast_enhancement(image_rgb, mask_rembg)

    if step_callback: step_callback("Adjusting lighting and tone balance")
    wbatm_img = white_balance_and_tone_mapping(gdce_img)

    if step_callback: step_callback("Refining edges with secondary mask")
    segmentation_img = rm_background_segmentation(wbatm_img)
    mask_gc, _ = rm_background_grabcut(wbatm_img)

    if step_callback: step_callback("Merging masks for clean subject edges")
    combo_mask = get_combo_mask(mask_rembg, mask_gc)

    if step_callback: step_callback("Compositing on white background")
    final_img = white_background(segmentation_img, combo_mask)

    if step_callback: step_callback("Adding shadow for realism")
    final_with_shadow = add_shadow(final_img[:, :, :3], combo_mask)

    return final_with_shadow.astype(np.uint8)
