import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from image_colorization import process_images, plot_histogram, plot_difference_map, plot_color_channels, plot_lab_channels
from skimage.color import rgb2lab, lab2rgb

st.set_page_config(page_title="SAR Image Colorization - Comparison", layout="wide")

st.title("Image Colorization of SAR Greyscale images")


if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'src' not in st.session_state:
    st.session_state.src = None
if 'gen' not in st.session_state:
    st.session_state.gen = None
if 'tar' not in st.session_state:
    st.session_state.tar = None

# Sidebar with information
st.sidebar.header("About")
st.sidebar.info(
    "This application demonstrates the capabilities of our SAR image colorization model. "
    "Upload a source (grayscale) image and a target (color) image to see the results and analysis."
)

st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1. Upload a source (grayscale) image
    2. Upload a target (color) image
    3. Click 'Process Images'
    4. Explore the results and analysis
    """
)


col1, col2 = st.columns(2)

with col1:
    src_image = st.file_uploader("Upload Source (Grayscale) Image", type=["png", "jpg", "jpeg"])

with col2:
    tar_image = st.file_uploader("Upload Target (Color) Image", type=["png", "jpg", "jpeg"])

if src_image and tar_image:
    if st.button("Process Images") or not st.session_state.processed:
        with st.spinner("Processing images..."):
            fig, ssim_value, psnr_value, src, gen, tar = process_images(src_image, tar_image)
            
            st.session_state.processed = True
            st.session_state.src = src
            st.session_state.gen = gen
            st.session_state.tar = tar
            
            st.pyplot(fig)
            
            st.subheader("Accuracy Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("SSIM", f"{ssim_value:.4f}")
                st.info("SSIM (Structural Similarity Index) measures the perceived similarity between two images. Values range from -1 to 1, where 1 indicates perfect similarity.")
            with col2:
                st.metric("PSNR", f"{psnr_value:.2f} dB")
                st.info("PSNR (Peak Signal-to-Noise Ratio) measures the quality of the reconstructed image compared to the original. Higher values indicate better quality.")

            # Color Histograms
            st.subheader("Color Histograms")
            st.write("Color histograms show the distribution of pixel intensities for each color channel (Red, Green, Blue).")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            plot_histogram((src + 1) / 2, ax1, "Source")
            plot_histogram((gen + 1) / 2, ax2, "Generated")
            plot_histogram((tar + 1) / 2, ax3, "Target")
            st.pyplot(fig)

            # Difference Map
            st.subheader("Difference Map")
            diff_fig, mean_diff = plot_difference_map((tar + 1) / 2, (gen + 1) / 2)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.pyplot(diff_fig)
            with col2:
                st.write("""
                The Difference Map shows areas where the generated image differs from the target image.
                - Brighter areas indicate larger differences.
                - Darker areas indicate smaller differences.
                - The color scale is square-root scaled to enhance visibility of small differences.
                """)
                st.metric("Mean Difference", f"{mean_diff:.4f}")
                st.info("The Mean Difference provides an overall measure of how much the generated image differs from the target. Lower values indicate better performance.")

            # Color Channels
            st.subheader("Color Channels")
            st.write("These visualizations show the intensity of each color channel (Red, Green, Blue) separately.")
            st.pyplot(plot_color_channels((gen + 1) / 2, "Generated Image Color Channels"))
            st.pyplot(plot_color_channels((tar + 1) / 2, "Target Image Color Channels"))

            # LAB Color Space
            st.subheader("LAB Color Space")
            st.write("""
            The LAB color space represents colors in a way that approximates human vision:
            - L: Lightness (0 = black, 100 = white)
            - a: Green-Red axis (negative = green, positive = red)
            - b: Blue-Yellow axis (negative = blue, positive = yellow)
            """)
            st.pyplot(plot_lab_channels((gen + 1) / 2, "Generated Image in LAB Color Space"))
            st.pyplot(plot_lab_channels((tar + 1) / 2, "Target Image in LAB Color Space"))

            # Image Download
            st.subheader("Download Images")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="Download Source Image",
                    data=src_image,
                    file_name="source_image.png",
                    mime="image/png"
                )
            with col2:
                # Convert generated image to bytes
                gen_img = ((gen + 1) / 2 * 255).astype(np.uint8)
                gen_img_bytes = gen_img.tobytes()
                st.download_button(
                    label="Download Generated Image",
                    data=gen_img_bytes,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            with col3:
                st.download_button(
                    label="Download Target Image",
                    data=tar_image,
                    file_name="target_image.png",
                    mime="image/png"
                )

    # Comparison Slider
    if st.session_state.processed:
        st.subheader("Image Comparison Slider")
        comparison_value = st.slider("Slide to compare original and generated images", 0.0, 1.0, 0.5)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(((st.session_state.src + 1) / 2) * (1 - comparison_value) + ((st.session_state.gen + 1) / 2) * comparison_value)
        ax.axis('off')
        st.pyplot(fig)
        st.write(f"Showing {comparison_value:.0%} of the generated image.")

else:
    st.info("Please upload both source and target images to begin the analysis.")


st.markdown("---")
st.markdown("Developed by Team Horizon | © 2024")
