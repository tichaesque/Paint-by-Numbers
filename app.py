import streamlit as st
import cv2
import numpy as np
import mixbox
import io

# --- Helper Functions ---
def adjust_saturation(img, scale):
    """Adjusts the saturation of an OpenCV image by a given scale factor."""
    if scale == 1.0:
        return img
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * scale
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
    hsv_img = hsv_img.astype(np.uint8)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def posterize_image(img, n_colors):
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, np.uint8(centers)

def clamp_to_mixable(b, g, r):
    """Forces an RGB color into the realistic bounds of physical paint using Mixbox."""
    latent = list(mixbox.rgb_to_latent((int(r), int(g), int(b))))
    latent[4], latent[5], latent[6] = 0.0, 0.0, 0.0  
    mix_r, mix_g, mix_b = mixbox.latent_to_rgb(latent)
    return int(mix_b), int(mix_g), int(mix_r)

def bgr_to_hex(b, g, r):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return b, g, r

def generate_mixing_key_array(colors):
    """Generates the mixing key and returns it as an RGB image array."""
    n_colors = len(colors)
    square_size, padding, bar_height, bar_spacing, text_space = 120, 40, 20, 10, 160  
    width = padding + n_colors * (square_size + text_space + padding)
    height = padding + square_size + padding + 4 * (bar_height + bar_spacing) + padding
    
    key_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    for i, color in enumerate(colors):
        b, g, r = color
        latent = mixbox.rgb_to_latent((int(r), int(g), int(b)))
        
        c = int(max(0.0, min(latent[0], 1.0)) * 100)
        y = int(max(0.0, min(latent[1], 1.0)) * 100)
        m = int(max(0.0, min(latent[2], 1.0)) * 100)
        w = int(max(0.0, min(latent[3], 1.0)) * 100)
        
        start_x = padding + i * (square_size + text_space + padding)
        
        cv2.rectangle(key_img, (start_x, padding), (start_x + square_size, padding + square_size), (int(b), int(g), int(r)), -1)
        cv2.rectangle(key_img, (start_x, padding), (start_x + square_size, padding + square_size), (0, 0, 0), 2)
        
        bar_start_y = padding + square_size + padding
        bars = [
            ('Cyan', c, (255, 255, 0)),    
            ('Magenta', m, (255, 0, 255)),    
            ('Yellow', y, (0, 255, 255)),
            ('White/Clear', w, (240,240,240))
        ]
        
        for j, (label, val, bar_color) in enumerate(bars):
            y_pos = bar_start_y + j * (bar_height + bar_spacing)
            fill_width = int((val / 100.0) * square_size)
            cv2.rectangle(key_img, (start_x, y_pos), (start_x + square_size, y_pos + bar_height), (200, 200, 200), 1)
            if fill_width > 0:
                cv2.rectangle(key_img, (start_x, y_pos), (start_x + fill_width, y_pos + bar_height), bar_color, -1)
            cv2.putText(key_img, f"{label}: {val}%", (start_x + square_size + 8, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return key_img

def generate_paint_by_numbers(labels2d, n_colors, img_shape):
    """Generates a paint-by-numbers image with outlined regions and numbers."""
    h, w = labels2d.shape
    pbn_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Draw outlines between different-colored regions
    for i in range(n_colors):
        mask = (labels2d == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pbn_img, contours, -1, (0, 0, 0), 1)

    # Determine font scale based on image size
    base_scale = min(h, w) / 800.0
    font_scale = max(0.3, min(base_scale * 0.5, 1.5))
    font_thickness = max(1, int(base_scale * 1.2))
    min_area = (min(h, w) / 50) ** 2  # skip tiny regions

    # Place numbers in each connected component
    for i in range(n_colors):
        mask = (labels2d == i).astype(np.uint8)
        num_components, comp_labels = cv2.connectedComponents(mask)

        for comp_id in range(1, num_components):
            comp_mask = (comp_labels == comp_id).astype(np.uint8)
            area = cv2.countNonZero(comp_mask)
            if area < min_area:
                continue

            # Use distance transform to find the point most interior to the region
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = max_loc

            label = str(i + 1)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            tx = cx - text_size[0] // 2
            ty = cy + text_size[1] // 2
            cv2.putText(pbn_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return pbn_img

def add_registration_marks(img, color=(0, 0, 0), length=25, thickness=3, offset=0):
    """Adds right-angle registration marks to the corners of an image."""
    marked_img = img.copy()
    h, w = marked_img.shape[:2]
    
    cv2.line(marked_img, (offset, offset), (offset + length, offset), color, thickness)
    cv2.line(marked_img, (offset, offset), (offset, offset + length), color, thickness)
    cv2.line(marked_img, (w - offset, offset), (w - offset - length, offset), color, thickness)
    cv2.line(marked_img, (w - offset, offset), (w - offset, offset + length), color, thickness)
    cv2.line(marked_img, (offset, h - offset), (offset + length, h - offset), color, thickness)
    cv2.line(marked_img, (offset, h - offset), (offset, h - offset - length), color, thickness)
    cv2.line(marked_img, (w - offset, h - offset), (w - offset - length, h - offset), color, thickness)
    cv2.line(marked_img, (w - offset, h - offset), (w - offset, h - offset - length), color, thickness)
    
    return marked_img

def append_guide_to_image(main_img, guide_img):
    """Vertically concatenates the main image and the mixing guide, centering them."""
    h1, w1 = main_img.shape[:2]
    h2, w2 = guide_img.shape[:2]
    
    target_w = max(w1, w2)
    
    if w1 < target_w:
        pad_left = (target_w - w1) // 2
        pad_right = target_w - w1 - pad_left
        main_padded = cv2.copyMakeBorder(main_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        main_padded = main_img
        
    if w2 < target_w:
        pad_left = (target_w - w2) // 2
        pad_right = target_w - w2 - pad_left
        guide_padded = cv2.copyMakeBorder(guide_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        guide_padded = guide_img
        
    separator = np.ones((40, target_w, 3), dtype=np.uint8) * 255
    return cv2.vconcat([main_padded, separator, guide_padded])

# --- Session State Callbacks ---
def update_palette_color(index):
    """Callback triggered when the user picks a new color."""
    picked_hex = st.session_state[f"color_picker_{index}"]
    b, g, r = hex_to_bgr(picked_hex)
    st.session_state.palette_bgr[index] = clamp_to_mixable(b, g, r)

# --- Streamlit Web UI ---
st.set_page_config(page_title="MIT Museum: Transforming Portraits", layout="wide")
st.title("🎨 MIT Museum: Transforming Portraits")
st.write("Upload an image to generate a customizable paint-by-numbers kit!")

if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.labels = None
    st.session_state.palette_bgr = []
    st.session_state.img_shape = None

st.sidebar.header("Settings")
n_colors = st.sidebar.slider("Number of Colors", min_value=2, max_value=8, value=4)
smoothing = st.sidebar.slider("Edge Smoothing", min_value=1, max_value=31, value=9)
saturation_scale = st.sidebar.slider("Saturation", min_value=0.0, max_value=3.0, value=1.0, step=0.1, 
                                     help="1.0 is original. < 1.0 mutes colors, > 1.0 boosts colors.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    opencv_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
    
    st.sidebar.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), caption="Working Image", use_container_width=True)
    
    if st.button("Generate Paint by Numbers"):
        with st.spinner('Running K-Means Clustering... This may take a few seconds.'):
            img = opencv_image.copy()

            # Resize to 1000px wide, maintaining aspect ratio
            h0, w0 = img.shape[:2]
            if w0 != 1000:
                new_w = 1000
                new_h = int(h0 * new_w / w0)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            img = adjust_saturation(img, saturation_scale)

            if smoothing % 2 == 0:
                smoothing += 1
            img = cv2.medianBlur(img, smoothing)

            raw_labels, raw_centers = posterize_image(img, n_colors)
            mixable_centers = [clamp_to_mixable(*c) for c in raw_centers]
            
            lum = [0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0] for c in mixable_centers]
            sorted_indices = np.argsort(lum)[::-1]
            
            st.session_state.palette_bgr = [mixable_centers[idx] for idx in sorted_indices]
            
            for i, color_bgr in enumerate(st.session_state.palette_bgr):
                st.session_state[f"color_picker_{i}"] = bgr_to_hex(*color_bgr)
            
            remapped_labels = np.zeros_like(raw_labels)
            for new_idx, old_idx in enumerate(sorted_indices):
                remapped_labels[raw_labels == old_idx] = new_idx
                
            st.session_state.labels = remapped_labels.flatten()
            st.session_state.img_shape = img.shape
            st.session_state.processed = True

    if st.session_state.processed:
        st.divider()
        st.subheader("🎨 Editable Palette")
        st.write("Click any color square below to edit the generated palette. The paint mixtures will instantly update!")
        
        cols = st.columns(len(st.session_state.palette_bgr))
        for i, color_bgr in enumerate(st.session_state.palette_bgr):
            current_hex = bgr_to_hex(*color_bgr)
            cols[i].color_picker(
                f"Layer {i+1}", 
                value=current_hex, 
                key=f"color_picker_{i}", 
                on_change=update_palette_color, 
                args=(i,)
            )

        labels2d = st.session_state.labels.reshape(st.session_state.img_shape[:2])
        posterized_full_image = np.zeros(st.session_state.img_shape, dtype=np.uint8)
        
        layers = []
        for i, color_bgr in enumerate(st.session_state.palette_bgr):
            mask = (labels2d == i)
            posterized_full_image[mask] = color_bgr
            
            color_layer = np.ones(st.session_state.img_shape, dtype=np.uint8) * 255
            color_layer[mask] = (0, 0, 0)
            hex_color = bgr_to_hex(*color_bgr)
            layers.append((hex_color, color_layer))

        mix_key = generate_mixing_key_array(st.session_state.palette_bgr)

        # st.subheader("1. Paint by Numbers")
        # pbn_img = generate_paint_by_numbers(labels2d, len(st.session_state.palette_bgr), st.session_state.img_shape)
        # st.image(cv2.cvtColor(pbn_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # marked_pbn = add_registration_marks(pbn_img)
        # combined_pbn = append_guide_to_image(marked_pbn, mix_key)

        # is_success_pbn, buffer_pbn = cv2.imencode(".png", combined_pbn)
        # if is_success_pbn:
        #     st.download_button(
        #         label="Download Paint by Numbers + Color Key",
        #         data=io.BytesIO(buffer_pbn),
        #         file_name="paint_by_numbers_with_guide.png",
        #         mime="image/png"
        #     )

        st.subheader("Full Posterized Preview")
        st.image(cv2.cvtColor(posterized_full_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        marked_full = add_registration_marks(posterized_full_image)
        combined_full = append_guide_to_image(marked_full, mix_key)

        is_success_preview, buffer_preview = cv2.imencode(".png", combined_full)
        if is_success_preview:
            st.download_button(
                label="Download Posterized Preview + Full Guide",
                data=io.BytesIO(buffer_preview),
                file_name="0_posterized_full_with_guide.png",
                mime="image/png"
            )

        st.subheader("Color Mixing Key")
        st.image(cv2.cvtColor(mix_key, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        st.subheader("Individual Color Layers")
        layer_cols = st.columns(len(layers))
        for i, (hex_code, layer_img) in enumerate(layers):
            with layer_cols[i]:
                st.image(cv2.cvtColor(layer_img, cv2.COLOR_BGR2RGB), caption=f"Layer {i+1} ({hex_code})")
                
                # --- NEW: Generate a guide with ONLY the current color ---
                single_color_key = generate_mixing_key_array([st.session_state.palette_bgr[i]])
                
                marked_layer = add_registration_marks(layer_img)
                
                # --- NEW: Append the single color key instead of the full mix_key ---
                combined_layer = append_guide_to_image(marked_layer, single_color_key)
                
                is_success, buffer = cv2.imencode(".png", combined_layer)
                if is_success:
                    st.download_button(
                        label=f"Download L{i+1} + Guide",
                        data=io.BytesIO(buffer),
                        file_name=f"layer_{i+1}_{hex_code}_with_guide.png",
                        mime="image/png"
                    )