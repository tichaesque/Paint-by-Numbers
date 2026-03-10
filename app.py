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
    
    # Convert to HSV and float32 for accurate math
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Multiply the saturation channel (index 1) by the scale factor
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * scale
    
    # Clip the values to ensure they stay within the valid 0-255 range
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
    
    # Convert back to uint8 and BGR
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
    latent[4], latent[5], latent[6] = 0.0, 0.0, 0.0  # Zero out unmixable digital residuals
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

# --- Session State Callbacks ---
def update_palette_color(index):
    """Callback triggered when the user picks a new color."""
    picked_hex = st.session_state[f"color_picker_{index}"]
    b, g, r = hex_to_bgr(picked_hex)
    st.session_state.palette_bgr[index] = clamp_to_mixable(b, g, r)

# --- Streamlit Web UI ---
st.set_page_config(page_title="Paint By Numbers Generator", layout="wide")
st.title("🎨 Paint By Numbers Generator")
st.write("Upload an image to generate an interactive, customizable paint-by-numbers kit.")

# Initialize Session State
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.labels = None
    st.session_state.palette_bgr = []
    st.session_state.img_shape = None

# Sidebar controls
st.sidebar.header("Settings")
n_colors = st.sidebar.slider("Number of Colors", min_value=2, max_value=4, value=4)
smoothing = st.sidebar.slider("Edge Smoothing", min_value=1, max_value=31, value=9, step=2)
saturation_scale = st.sidebar.slider("Saturation", min_value=0.0, max_value=3.0, value=1.0, step=0.1, 
                                     help="1.0 is original. < 1.0 mutes colors, > 1.0 boosts colors.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    st.sidebar.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    
    # --- Generate Button (Runs heavy K-Means) ---
    if st.button("Generate Paint by Numbers"):
        with st.spinner('Running K-Means Clustering... This may take a few seconds.'):
            img = opencv_image.copy()
            
            # Apply the new saturation scaling
            img = adjust_saturation(img, saturation_scale)

            if smoothing % 2 == 0:
                smoothing += 1
            img = cv2.medianBlur(img, smoothing)

            raw_labels, raw_centers = posterize_image(img, n_colors)
            
            # Clamp initial centers to mixable colors
            mixable_centers = [clamp_to_mixable(*c) for c in raw_centers]
            
            # Sort by luminance (light to dark)
            lum = [0.299 * c[2] + 0.587 * c[1] + 0.114 * c[0] for c in mixable_centers]
            sorted_indices = np.argsort(lum)[::-1]
            
            # Save the sorted palette to session state
            st.session_state.palette_bgr = [mixable_centers[idx] for idx in sorted_indices]
            
            # --- NEW FIX: Force color picker widgets to update their internal state ---
            for i, color_bgr in enumerate(st.session_state.palette_bgr):
                st.session_state[f"color_picker_{i}"] = bgr_to_hex(*color_bgr)
            
            # Remap the labels so they match the newly sorted palette index
            remapped_labels = np.zeros_like(raw_labels)
            for new_idx, old_idx in enumerate(sorted_indices):
                remapped_labels[raw_labels == old_idx] = new_idx
                
            st.session_state.labels = remapped_labels.flatten()
            st.session_state.img_shape = img.shape
            st.session_state.processed = True

    # --- Interactive UI (Runs instantly) ---
    if st.session_state.processed:
        st.divider()
        st.subheader("🎨 Editable Palette")
        st.write("Click any color square below to edit the generated palette. The paint mixtures will instantly update!")
        
        # Display Color Pickers
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

        # Build Composite Image from State
        labels2d = st.session_state.labels.reshape(st.session_state.img_shape[:2])
        posterized_full_image = np.zeros(st.session_state.img_shape, dtype=np.uint8)
        
        layers = []
        for i, color_bgr in enumerate(st.session_state.palette_bgr):
            mask = (labels2d == i)
            posterized_full_image[mask] = color_bgr
            
            # Generate Individual Layer Image
            color_layer = np.ones(st.session_state.img_shape, dtype=np.uint8) * 255
            color_layer[mask] = color_bgr
            hex_color = bgr_to_hex(*color_bgr)
            layers.append((hex_color, color_layer))

        # Render Results
        st.subheader("1. Full Posterized Preview")
        st.image(cv2.cvtColor(posterized_full_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        is_success_preview, buffer_preview = cv2.imencode(".png", posterized_full_image)
        if is_success_preview:
            st.download_button(
                label="Download Posterized Preview",
                data=io.BytesIO(buffer_preview),
                file_name="0_posterized_full.png",
                mime="image/png"
            )
        
        st.subheader("2. Color Mixing Key")
        mix_key = generate_mixing_key_array(st.session_state.palette_bgr)
        st.image(cv2.cvtColor(mix_key, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        is_success_key, buffer_key = cv2.imencode(".png", mix_key)
        if is_success_key:
            st.download_button(
                label="Download Pigment Mixing Key",
                data=io.BytesIO(buffer_key),
                file_name="mixing_key.png",
                mime="image/png"
            )
        
        st.subheader("3. Individual Color Layers")
        layer_cols = st.columns(len(layers))
        for i, (hex_code, layer_img) in enumerate(layers):
            with layer_cols[i]:
                st.image(cv2.cvtColor(layer_img, cv2.COLOR_BGR2RGB), caption=f"Layer {i+1} ({hex_code})")
                
                is_success, buffer = cv2.imencode(".png", layer_img)
                if is_success:
                    st.download_button(
                        label=f"Download L{i+1}",
                        data=io.BytesIO(buffer),
                        file_name=f"layer_{i+1}_{hex_code}.png",
                        mime="image/png"
                    )