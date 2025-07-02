import os
import base64
import tempfile

from langchain_core.messages import HumanMessage

from src.state_models import (FileProcessingState, MainState,
                              FileSource)
from src.utils.download_helper import download_file_from_url
from src.utils.llm_factory import get_llm
from src.file_processing_sub_graph import file_processing_graph
from src.utils.logger import get_logger
from src.utils.image_converter import (
    convert_to_standard_jpeg, cleanup_temp_file
)
from config import OPENAI_API_KEY, COST_SAVING_OPENAI_MODEL_NAME

# Initialize logger at module level
logger = get_logger("nodes")


def aggregate_content(state: MainState) -> MainState:
    """Aggregates text and images from all processed files."""
    logger.info("Aggregating content from all files...")

    aggregated_texts = []
    all_images = []

    processing_results = state.processing_results or []
    for res in processing_results:
        # Use original_path_or_url for display if available,
        # otherwise file_path
        display_name = res.original_path_or_url or os.path.basename(
            res.file_path or 'unknown')
        if res.extracted_text:
            aggregated_texts.append(f"--- Content from {display_name} ---\n" +
                                    f"{res.extracted_text}")
        if res.extracted_images:
            # Convert any remaining images to standard JPEG format
            for img_bytes in res.extracted_images:
                try:
                    logger.info(
                        f"Processing image from {display_name} " +
                        f"(size: {len(img_bytes)} bytes)"
                    )

                    # Save image bytes to temporary file for conversion
                    temp_img_path = tempfile.mktemp(suffix='.png')
                    with open(temp_img_path, 'wb') as f:
                        f.write(img_bytes)

                    logger.debug(f"Saved temporary image to: {temp_img_path}")

                    # Convert to standard JPEG format
                    converted_path = convert_to_standard_jpeg(
                        temp_img_path,
                        max_resolution=512,
                        quality=70
                    )

                    if converted_path:
                        # Read the converted JPEG image
                        with open(converted_path, "rb") as f:
                            converted_bytes = f.read()
                            all_images.append(converted_bytes)

                        logger.info("Successfully converted image from " +
                                    f"{display_name} to JPEG " +
                                    f"(original: {len(img_bytes)} bytes, " +
                                    f"converted: {len(converted_bytes)} bytes)"
                                    )

                        # Clean up temporary files
                        cleanup_temp_file(converted_path)
                        cleanup_temp_file(temp_img_path)
                    else:
                        # Fallback to original image if conversion fails
                        all_images.append(img_bytes)
                        cleanup_temp_file(temp_img_path)
                        logger.warning(
                            f"Failed to convert image from {display_name}, " +
                            "using original"
                        )

                except Exception as e:
                    # If conversion fails, use original image
                    all_images.append(img_bytes)
                    logger.warning(
                        f"Error converting image from {display_name}: {e}, " +
                        "using original"
                    )

    state.aggregated_text = "\n\n".join(aggregated_texts)
    state.all_extracted_images = all_images
    logger.info(f"Aggregated Text Length: {len(state.aggregated_text or '')}")
    logger.info(f"Aggregated Image Count: {len(all_images)}")
    logger.debug("Aggregated Text Snippet: %s...",
                 (state.aggregated_text or '')[:500])
    return state


def generate_answer(state: MainState) -> MainState:
    """Generates a final answer using the aggregated content."""
    logger.info("Generating final answer...")

    llm = get_llm("openai",
                  OPENAI_API_KEY,
                  COST_SAVING_OPENAI_MODEL_NAME)

    aggregated_content = state.aggregated_text or ''
    all_extracted_images = state.all_extracted_images or []

    if not aggregated_content and not all_extracted_images:
        state.answer = "I'm sorry, but I couldn't extract any content from " +\
            "the provided document(s)."
        logger.warning("Generated answer: No content extracted.")
        return state

    # Constructing the prompt with text content
    question = state.question or "Please analyze the provided content."
    prompt_content = [{
        "type": "text",
        "text": "You are an expert analyst. Based ONLY on the " +
        "content from the provided documents, answer the " +
        f"following question. Question: {question}\n\n{aggregated_content}"
    }]

    # Adding image content if available, with correct base64 encoding
    if all_extracted_images:
        logger.info(f"Processing {len(all_extracted_images)} images for LLM")
        for i, img_bytes in enumerate(all_extracted_images):
            logger.info(
                f"Image {i}: {len(img_bytes)} bytes, " +
                f"first 20 bytes: {img_bytes[:20].hex()}"
            )
            encoded_image = base64.b64encode(img_bytes).decode('utf-8')
            prompt_content.append({
                "type": "image_url",
                # For most LLMs, it's image/jpeg, but you might need to infer
                # from actual image type
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
            logger.info(f"Added image {i} to prompt " +
                        f"(encoded length: {len(encoded_image)})")

    messages = [HumanMessage(content=prompt_content)]

    logger.info("Sending prompt to LLM. Prompt snippet: " +
                f"{prompt_content[0]['text'][:500]}...")
    response = llm.invoke(messages)
    state.answer = response.content
    logger.info(f"Generated answer: {state.answer[:200]}...")
    return state


def process_files_node(state: MainState) -> MainState:
    """Processes each file input sequentially, handling URLs by
    downloading them.
    """
    logger.info("Processing files...")

    processing_results = []
    file_inputs = state.file_inputs or []

    for file_input in file_inputs:
        original_path_or_url = file_input.file_path_or_url
        file_source = file_input.file_source
        local_file_path = None

        try:
            if file_source == FileSource.URL:
                logger.info(f"Processing URL: {original_path_or_url}")
                local_file_path = download_file_from_url(original_path_or_url)
            else:
                logger.info(f"Processing local file: {original_path_or_url}")
                local_file_path = original_path_or_url

            subgraph_input = {
                "original_path_or_url": original_path_or_url,
                "file_path": local_file_path
            }
            result = file_processing_graph.invoke(subgraph_input)
            processing_results.append(result)

        except Exception as e:
            logger.error(f"Error processing {file_source.value} " +
                         f"{original_path_or_url}: {e}")
            error_result = FileProcessingState(
                original_path_or_url=original_path_or_url,
                file_path=local_file_path if local_file_path else "N/A",
                file_type="error",
                extracted_text="Error processing file from " +
                f"{file_source.value}: {e}",
                extracted_images=[]
            )
            processing_results.append(error_result)
        finally:
            # Clean up temporary file if it was downloaded from a URL
            if file_source == FileSource.URL and local_file_path and \
                    os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                    logger.info(
                        f"Cleaned up temporary file: {local_file_path}"
                        )
                except Exception as e:
                    logger.error("Error cleaning up temporary file " +
                                 f"{local_file_path}: {e}")

    logger.info("Processing completed. Results count: " +
                f"{len(processing_results)}")
    state.processing_results = processing_results
    return state
