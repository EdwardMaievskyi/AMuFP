#!/usr/bin/env python3
"""
Example usage of the Pydantic state models for LangGraph.
This demonstrates the improved data validation and manipulation capabilities.
"""

from src.state_models import (
    FileSource, FileMetaData, FileProcessingState, MainState
)


def main():
    """Demonstrate the usage of Pydantic state models."""
    
    # Example 1: Creating FileMetaData with validation
    print("=== Example 1: FileMetaData Validation ===")
    
    try:
        # Empty state (for integration with larger state models)
        empty_file = FileMetaData()
        print(f"✓ Empty file metadata: {empty_file}")
        
        # Valid local file
        local_file = FileMetaData(
            file_path_or_url="/path/to/document.pdf",
            file_source=FileSource.LOCAL_PATH
        )
        print(f"✓ Valid local file: {local_file}")
        
        # Valid URL
        url_file = FileMetaData(
            file_path_or_url="https://example.com/document.pdf",
            file_source=FileSource.URL
        )
        print(f"✓ Valid URL file: {url_file}")
        
        # Invalid empty path
        try:
            FileMetaData(
                file_path_or_url="",
                file_source=FileSource.LOCAL_PATH
            )
        except ValueError as e:
            print(f"✗ Caught validation error: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Creating FileProcessingState
    print("\n=== Example 2: FileProcessingState ===")
    
    try:
        # Empty state (for integration with larger state models)
        empty_state = FileProcessingState()
        print(f"✓ Empty processing state: {empty_state}")
        
        processing_state = FileProcessingState(
            original_path_or_url="/path/to/document.pdf",
            file_path="/tmp/processed/document.pdf",
            file_type="pdf",
            extracted_text="This is the extracted text from the document.",
            extracted_images=[b"fake_image_data"]
        )
        print(f"✓ Processing state created: {processing_state}")
        print(f"  File type normalized: {processing_state.file_type}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Creating and manipulating MainState
    print("\n=== Example 3: MainState Manipulation ===")
    
    try:
        # Create empty main state (for integration with larger state models)
        empty_main_state = MainState()
        print(f"✓ Empty main state: {empty_main_state}")
        
        # Create main state
        main_state = MainState(
            question="What is the main topic of the document?",
            model_provider="openai"
        )
        print(f"✓ Main state created: {main_state.question}")
        
        # Add file inputs
        file1 = FileMetaData(
            file_path_or_url="/path/to/doc1.pdf",
            file_source=FileSource.LOCAL_PATH
        )
        file2 = FileMetaData(
            file_path_or_url="https://example.com/doc2.pdf",
            file_source=FileSource.URL
        )
        
        main_state.add_file_input(file1)
        main_state.add_file_input(file2)
        print(f"✓ Added {len(main_state.file_inputs)} file inputs")
        
        # Add processing results
        result1 = FileProcessingState(
            original_path_or_url="/path/to/doc1.pdf",
            file_path="/tmp/processed/doc1.pdf",
            file_type="pdf",
            extracted_text="Document 1 content here."
        )
        
        main_state.add_processing_result(result1)
        print("✓ Added processing result")
        
        # Update aggregated text
        main_state.update_aggregated_text("Combined content from all documents.")
        print(f"✓ Updated aggregated text: {main_state.aggregated_text[:50]}...")
        
        # Set final answer
        main_state.set_answer("The main topic is data processing.")
        print(f"✓ Set answer: {main_state.answer}")
        
        # Display final state summary
        print("\n=== Final State Summary ===")
        print(f"Question: {main_state.question}")
        print(f"Files to process: {len(main_state.file_inputs)}")
        print(f"Processing results: "
              f"{len(main_state.processing_results)}")
        print(f"Model provider: {main_state.model_provider}")
        print(f"Answer: {main_state.answer}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: JSON serialization/deserialization
    print("\n=== Example 4: JSON Serialization ===")
    
    try:
        # Create a simple state
        state = MainState(question="Test question")
        
        # Convert to JSON
        json_data = state.model_dump_json()
        print(f"✓ Serialized to JSON: {json_data[:100]}...")
        
        # Parse from JSON
        parsed_state = MainState.model_validate_json(json_data)
        print(f"✓ Parsed from JSON: {parsed_state.question}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 