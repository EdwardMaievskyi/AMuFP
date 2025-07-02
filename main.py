from src.state_models import FileMetaData, MainState
from src.graph_builder import app
from src.utils.logger import init_logging, get_logger


if __name__ == "__main__":
    # Initialize logging
    init_logging(level="CRITICAL", console_output=True)
    logger = get_logger("main")

    logger.info("Starting AMuFP application")

    # Create file inputs using the new Pydantic models
    # Example file paths for testing:
    # - "/home/mayak/Documents/AMuFP/AMuFP/"
    #   "EM_empowering_data_enthusiasts_with_open_source_tools_test.odp"
    # - "https://thunderdungeon.com/wp-content/uploads/2024/09/it-memes-6.jpg"
    # - "/home/mayak/Documents/AMuFP/AMuFP/test_pdf.pdf"
    # - "/home/mayak/Documents/AMuFP/AMuFP/face.svg"
    file_inputs = [
        FileMetaData(
            file_path_or_url="https://thunderdungeon.com/wp-content/uploads/2024/09/it-memes-6.jpg"
        ),
        FileMetaData(
            file_path_or_url="/home/mayak/Documents/AMuFP/AMuFP/face.svg"
        )
    ]

    inputs = {
        "question": "Describe images",
        "file_inputs": file_inputs,
        "model_provider": "openai"
    }

    logger.info("Invoking graph with inputs: %s", inputs)
    result = app.invoke(inputs)

    # Convert the result back to a MainState model for proper access
    final_state = MainState.model_validate(result)
    logger.info("Generated answer: %s", final_state.answer)
    print(final_state.answer)
