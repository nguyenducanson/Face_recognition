import gradio as gr

from services import (
    add_data,
    delete_data,
    process_camera_input,
)


# Gradio interface for the first tab (camera input and face detection)
def camera_tab():
    return gr.Interface(
        fn=process_camera_input,
        inputs=gr.Video(sources=["webcam"], format="mp4", width=200),
        outputs=gr.Image(),
    )


# Gradio interface for the second tab (update face database)
def database_tab():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(label="Enter Account")
                image_input = gr.Image(label="Face input")
                add_button = gr.Button("Add Face to Database")
            with gr.Column():
                output_text = gr.Textbox(label="Output")
                delete_button = gr.Button("Delete Face from Database")

        add_button.click(add_data, inputs=[name_input, image_input], outputs=output_text)
        delete_button.click(delete_data, inputs=name_input, outputs=output_text)
    return interface


if __name__ == "__main__":
    # Combine both tabs into a single Gradio app
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.TabItem("Face Detection and Matching"):
                camera_tab()

            with gr.TabItem("Update Face Database"):
                database_tab()

    # Launch the app
    app.launch()
