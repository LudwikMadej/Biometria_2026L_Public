import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Backend Imports ---
from backend import *

# --- App Config ---
st.set_page_config(page_title="Image Processing Dashboard", layout="wide")

# --- Session State Initialization ---
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "stack_changes" not in st.session_state:
    st.session_state.stack_changes = False


# --- Helper Functions ---
def load_uploaded_file(uploaded_file):
    image = load_image_to_numpy(uploaded_file)
    st.session_state.original_image = image.copy()
    st.session_state.current_image = image.copy()


def reset_image():
    if st.session_state.original_image is not None:
        st.session_state.current_image = st.session_state.original_image.copy()


def apply_transform(func, *args, **kwargs):
    """Applies a function to the image based on the stack_changes toggle."""
    if st.session_state.current_image is None:
        st.warning("Please upload an image first.")
        return

    source_img = (
        st.session_state.current_image
        if st.session_state.stack_changes
        else st.session_state.original_image
    )

    # Apply function and update current image
    st.session_state.current_image = func(source_img, *args, **kwargs)


def get_image_download_buffer(img_array):
    """Converts numpy array to a downloadable image buffer."""
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if "show_projections" not in st.session_state:
    st.session_state.show_projections = False


def switch_projections(show: bool):
    st.session_state.show_projections = show


# --- Main UI Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.header("Image Viewer")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Only load if it's a new file to prevent resetting on every interaction
        if (
            "last_uploaded" not in st.session_state
            or st.session_state.last_uploaded != uploaded_file.name
        ):
            load_uploaded_file(uploaded_file)
            st.session_state.last_uploaded = uploaded_file.name

    # 2. Controls & Image Display
    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, width="stretch")

        controls_col1, controls_col2, controls_col3 = st.columns(3)
        with controls_col1:
            st.button("Restart Image", on_click=reset_image, width="content")
        with controls_col2:
            st.checkbox(
                "Stack changes",
                key="stack_changes",
                help="If checked, new edits apply to the already edited image.",
            )
        with controls_col3:
            img_buffer = get_image_download_buffer(st.session_state.current_image)
            st.download_button(
                label="Save Image",
                data=img_buffer,
                file_name="edited_image.png",
                mime="image/png",
                width="content",
            )

        # 3. Image Info Table
        st.subheader("Image Information")
        img = st.session_state.current_image
        info_data = {
            "Width": img.shape[1],
            "Height": img.shape[0],
            "Channels": img.shape[2] if img.ndim == 3 else 1,
            "Data Type": str(img.dtype),
            "Min Pixel Brightness": float(np.min(img)),
            "Max Pixel Brightness": float(np.max(img)),
            "Mean Pixel Brightness": round(float(np.mean(img)), 2),
            "Median Pixel Brightness": round(float(np.median(img)), 2),
            "Contrast as a Relative Dynamic": round(
                (np.max(img) - np.min(img)) / 255, 2
            ),
        }
        st.table(pd.Series(info_data, name="Property").astype(str))

        # 4. Histograms
        if st.checkbox("Show Histograms"):
            figs = plot_brightness_histograms(img, normalize=False)

            # Display based on returned dictionary keys (Grayscale vs RGB)
            if "combined" in figs and len(figs) == 1:
                st.pyplot(figs["combined"])
            else:
                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    st.pyplot(figs["red"])
                    st.pyplot(figs["blue"])
                with h_col2:
                    st.pyplot(figs["green"])
                    st.pyplot(figs["combined"])

with col_right:
    st.header("Tools & Filters")
    (
        tab_pixels,
        tab_pixels2,
        tab_filters,
        tab_edge_detection,
        tab_custom_filters,
        tab_morphology,
        tab_stats,
    ) = st.tabs(
        [
            "Pixel Functions",
            "Pixel Functions 2",
            "Filters",
            "Edge Detection",
            "Custom Filters",
            "Morphological operations",
            "Projections",
        ]
    )

    # --- TAB: Pixel Functions ---
    with tab_pixels:
        with st.form("brightness_settings"):
            st.markdown("**Adjust Brightness**")
            beta = st.slider(
                "Beta (Brightness Offset)",
                min_value=-255.0,
                max_value=255.0,
                value=0.0,
                step=5.0,
            )
            submitted = st.form_submit_button(
                "Apply Brightness",
                key="btn_brightness",
            )

            if submitted:
                apply_transform(adjust_brightness, beta)
                st.rerun()

        with st.form("contrast_settings"):
            st.markdown("**Adjust Contrast**")
            alpha = st.slider(
                "Alpha (Contrast Multiplier)",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.2,
            )
            submitted = st.form_submit_button(
                "Apply Contrast",
                key="btn_contrast",
            )

            if submitted:
                apply_transform(adjust_contrast, alpha)
                st.rerun()

        with st.form("binarization_settings"):
            st.markdown("**Binarize Image**")
            threshold = st.slider("Threshold", min_value=0, max_value=255, value=127)
            submitted = st.form_submit_button(
                "Apply Binarization",
                key="btn_binarize",
            )

            if submitted:
                apply_transform(binarize_image, threshold)
                st.rerun()

        with st.container(border=True):
            st.markdown("**Grayscale & Invert**")
            col_g, col_i, col_e = st.columns(3)
            with col_g:
                st.button(
                    "Convert to Grayscale",
                    on_click=apply_transform,
                    args=[convert_to_grayscale],
                    width="content",
                )
            with col_i:
                st.button(
                    "Invert Colors",
                    on_click=apply_transform,
                    args=[invert_image],
                    width="content",
                )
            with col_e:
                st.button(
                    "Equalize Histogram",
                    on_click=apply_transform,
                    args=[equalize_histograms_grayscale],
                    width="content",
                )

    # --- TAB: Pixel Functions 2 ---
    with tab_pixels2:
        with st.form("log_form"):
            st.markdown("**Logarithmic Transformation**")

            submitted_log = st.form_submit_button("Apply Log")

            if submitted_log:
                apply_transform(log_image)
                st.rerun()

        with st.form("exp_form"):
            st.markdown("**Exponential Transformation**")
            alpha = st.slider(
                "Alpha (Gamma)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Values < 1 brighten the image, values > 1 darken it.",
            )

            submitted_exp = st.form_submit_button("Apply Exponential")

            if submitted_exp:
                apply_transform(exponentiate_image, alpha)
                st.rerun()

        with st.form("stretch_form"):
            st.markdown("**Brightness Stretching**")
            n1, n2 = st.slider("Set Brightness Boundaries", 0, 255, (0, 255))
            submitted_stretch = st.form_submit_button("Apply Stretching")

            if submitted_stretch:
                apply_transform(stretch_brightness, n1, n2)
                st.rerun()

    # --- TAB: Filter Functions ---
    with tab_filters:
        with st.form("laplacian_sharpen_form"):
            st.markdown("**Sharpening With Laplacian**")

            submitted_log = st.form_submit_button("Apply Laplacian Sharpening")

            if submitted_log:
                apply_transform(laplacian_operator, True)
                st.rerun()

        with st.form("averaging_settings"):
            st.markdown("**Averaging Filter (Box Blur)**")
            custom_middle = st.number_input(
                "Custom filter middle value",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.2,
            )

            submitted = st.form_submit_button(
                "Apply Averaging Filter",
                key="btn_avg_blur",
            )

            if submitted:
                apply_transform(averaging_filter, custom_middle)
                st.rerun()

        with st.form("gaussian_settings"):
            st.markdown("**Gaussian Operator**")
            b_val = st.number_input(
                "Weight", min_value=1, max_value=10, value=4, step=1
            )
            submitted = st.form_submit_button(
                "Apply Gaussian Filter",
                key="btn_gauss",
            )

            if submitted:
                apply_transform(gaussian_operator, b_val)
                st.rerun()
        with st.form("sharpening_settings"):
            submitted = st.markdown("**Sharpening Filter**")
            weight = st.number_input(
                "Center Weight", min_value=5.0, max_value=25.0, value=5.0, step=1.0
            )
            submitted = st.form_submit_button(
                "Apply Sharpening",
                key="btn_sharpen",
            )

            if submitted:
                apply_transform(sharpening_filter, weight)
                st.rerun()

    # --- TAB: Edge Detection ---
    with tab_edge_detection:
        with st.container(border=True):
            st.markdown("**Edge Detection Operators**")

            col_l, col_p, col_r = st.columns(3)
            col_sc, col_so, _ = st.columns(3)

            with col_l:
                st.button(
                    "Laplacian Operator",
                    on_click=apply_transform,
                    args=[laplacian_operator],
                    use_container_width=True,
                )

            with col_p:
                st.button(
                    "Prewitt Operator",
                    on_click=apply_transform,
                    args=[prewitt_operator],
                    use_container_width=True,
                )

            with col_r:
                st.button(
                    "Roberts Cross",
                    on_click=apply_transform,
                    args=[roberts_cross],
                    use_container_width=True,
                )

            with col_sc:
                st.button(
                    "Scharr Operator",
                    on_click=apply_transform,
                    args=[scharr_operator],
                    use_container_width=True,
                )

            with col_so:
                st.button(
                    "Sobel Operator",
                    on_click=apply_transform,
                    args=[sobel_operator],
                    use_container_width=True,
                )

    # --- TAB: Custom Filters ---
    with tab_custom_filters:
        with st.form("custom_kernel_form"):
            st.markdown("**Custom 3x3 Convolution Kernel**")
            st.info("Enter values for the 3x3 matrix to apply a custom filter.")

            k_cols = st.columns(3)

            kernel_values = []
            for row in range(3):
                for col in range(3):
                    with k_cols[col]:
                        val = st.number_input(
                            f"r{row}c{col}",
                            min_value=-15,
                            max_value=15,
                            value=0,
                            step=1,
                            label_visibility="collapsed",
                            key=f"k_{row}_{col}",
                        )
                        kernel_values.append(val)

            custom_kernel = np.array(kernel_values).reshape(3, 3)

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                norm = st.checkbox(
                    "Normalize Output",
                    value=True,
                    help="Scales result to [0, 255]. For most applications set to True.",
                )
            with col_opt2:
                gray = st.checkbox("Grayscale first", value=False)

            submitted_custom = st.form_submit_button(
                "Apply Custom Kernel", use_container_width=True
            )

            if submitted_custom:
                apply_transform(apply_filter, custom_kernel, norm, gray)
                st.rerun()

    # --- TAB: Morphological Operations ---
    with tab_morphology:
        with st.container(border=True):
            st.markdown("**Morphological Operations**")

            col_1, col_2, col_3 = st.columns(3)
            col_4, col_5, col_6 = st.columns(3)
            col_7, _, _ = st.columns(3)

            with col_1:
                st.button(
                    "Erode",
                    on_click=apply_transform,
                    args=[erode],
                    use_container_width=True,
                )

            with col_2:
                st.button(
                    "Dilate",
                    on_click=apply_transform,
                    args=[dilate],
                    use_container_width=True,
                )

            with col_3:
                st.button(
                    "Opening",
                    on_click=apply_transform,
                    args=[opening],
                    use_container_width=True,
                )

            with col_4:
                st.button(
                    "Closing",
                    on_click=apply_transform,
                    args=[closing],
                    use_container_width=True,
                )

            with col_5:
                st.button(
                    "Morphological Gradient",
                    on_click=apply_transform,
                    args=[morphological_gradient],
                    use_container_width=True,
                )

            with col_6:
                st.button(
                    "Top Hat",
                    on_click=apply_transform,
                    args=[top_hat],
                    use_container_width=True,
                )

            with col_7:
                st.button(
                    "Black Hat",
                    on_click=apply_transform,
                    args=[black_hat],
                    use_container_width=True,
                )

        # --- Form: Skeletonization ---
        with st.form("skeletonize_settings"):
            st.markdown("**Skeletonization (Lantuejoul's Algorithm)**")
            skel_thresh = st.slider(
                "Binarization Threshold", 0, 255, 128, key="skel_thr"
            )

            submitted_skel = st.form_submit_button(
                "Apply Skeletonization", use_container_width=True
            )

            if submitted_skel:
                apply_transform(skeletonize, None, skel_thresh)
                st.rerun()

        # --- Form: Morphological Reconstruction ---
        with st.form("reconstruction_settings"):
            st.markdown("**Morphological Reconstruction**")
            col_rec1, col_rec2 = st.columns(2)

            with col_rec1:
                recon_ops_dict = {
                    "Clear border": "clear_border",
                    "Fill holes": "fill_holes",
                    "H dome": "h_dome",
                }
                recon_op = st.selectbox(
                    "Operation Type",
                    ["Clear border", "Fill holes", "H dome"],
                    help="Select the specific reconstruction algorithm.",
                )
            with col_rec2:
                h_val = st.number_input(
                    "H value (H dome only)", min_value=1, max_value=255, value=10
                )

            rec_thresh = st.slider("Binarization Threshold", 0, 255, 128, key="rec_thr")

            submitted_rec = st.form_submit_button(
                "Apply Reconstruction", use_container_width=True
            )
            if submitted_rec:
                apply_transform(
                    morphological_reconstruction,
                    recon_ops_dict[recon_op],
                    h_val,
                    None,
                    rec_thresh,
                )
                st.rerun()

        # --- Form: Hit-or-Miss ---
        with st.form("hit_or_miss_settings"):
            st.markdown("**Hit-or-Miss Transform**")
            st.info(
                "Pattern values: 1 = Hit (Object), -1 = Miss (Background), 0 = Ignore"
            )

            hom_cols = st.columns(3)
            hom_values = []
            for row in range(3):
                for col in range(3):
                    with hom_cols[col]:
                        val = st.selectbox(
                            f"p{row}{col}",
                            [-1, 0, 1],
                            index=1,
                            key=f"hom_{row}_{col}",
                            label_visibility="collapsed",
                        )
                        hom_values.append(val)

            hom_pattern = np.array(hom_values).reshape(3, 3)
            hom_thresh = st.slider("Binarization Threshold", 0, 255, 128, key="hom_thr")

            submitted_hom = st.form_submit_button(
                "Apply Hit-or-Miss", use_container_width=True
            )
            if submitted_hom:
                apply_transform(hit_or_miss, hom_pattern, hom_thresh)
                st.rerun()

    # --- TAB: Projections & Stats ---
    with tab_stats:
        with st.container(border=True):
            st.markdown("**Plot Projections**")
            st.caption("Displays image projections (e.g., sum of pixels along axes).")
            st.button(
                (
                    "Generate Projections"
                    if not st.session_state.show_projections
                    else "Hide Projections"
                ),
                on_click=switch_projections,
                args=[not st.session_state.show_projections],
                key="btn_proj",
            )

        if st.session_state.show_projections:
            if st.session_state.current_image is not None:
                st.pyplot(plot_projections(st.session_state.current_image)["combined"])
