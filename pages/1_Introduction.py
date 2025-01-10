import streamlit as st

st.set_page_config(
    page_title="Introduction",
)

st.title("Introduction")
st.markdown("We’re delving into the intersection of AI and healthcare—an area where technology truly has the power to save lives. Our focus is on breast cancer detection using histopathological images from the BreaKHis dataset. This is not just about exploring a dataset but tackling one of the most pressing global health challenges with some of the most innovative tools in artificial intelligence.")

st.markdown("Breast cancer diagnosis relies heavily on histopathology, where tissues are analyzed at a cellular level. However, this process is time-consuming, highly specialized, and prone to variability in results. That’s where machine learning models step in, with the potential to automate and enhance diagnostic accuracy. Our team’s mission has been to explore how Vision Transformers (ViTs) and their advanced variants perform in this critical application, comparing them with AlexNet, a traditional yet impactful CNN.")

st.markdown("Vision Transformers are revolutionary in the AI world. They process images by breaking them into smaller patches, embedding positional information, and analyzing them using transformer blocks. This approach allows them to capture intricate patterns and relationships that are particularly relevant when dealing with complex histopathological images. A step further, the Swin Transformer optimizes this process by focusing on smaller, localized regions, making it more efficient while retaining accuracy.")

st.markdown("Now, let’s address a unique aspect of our work—the multiclass classification problem. The BreaKHis dataset includes eight distinct classes: adenosis, ductal carcinoma, fibroadenoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma, phyllodes tumor, and tubular adenoma. This diversity makes the task more complex. Each of these classes represents unique histopathological characteristics, so our model needed to identify these subtle differences accurately. Achieving this required a balanced dataset. The original data was skewed, with some classes overrepresented. To overcome this, we implemented a balancing strategy during preprocessing, ensuring fair representation for all classes.")

st.markdown("Our work aims to bridge the gap between cutting-edge AI and clinical application. By comparing ViTs, Swin Transformers, and AlexNet, we’ve been able to identify strengths and limitations, providing insights into which models are most suitable for such critical tasks. The potential here is enormous: imagine AI systems that can not only assist but actively empower healthcare professionals, making advanced diagnostics accessible even in resource-constrained settings.In the end, this is more than just a technical exercise. It’s about using technology to make a tangible difference, to innovate for a healthier future. And today, we’re excited to share the journey that brought us here.")


st.sidebar.image("uos.png", use_container_width=True, caption="Breast Cancer Detection")
