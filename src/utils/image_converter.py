"""
Image format conversion utilities for AMuFP.

This module provides functionality to convert unsupported image formats
to formats that LLMs can process (PNG, JPEG, GIF, WebP).
"""

import os
import tempfile
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
import subprocess

from PIL import Image
import cairosvg

from src.utils.logger import get_logger

# Initialize logger at module level
logger = get_logger("image_converter")

# Comprehensive MIME type mapping for all image formats
IMAGE_MIME_TYPES = {
    # Raster formats
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'tiff': 'image/tiff',
    'tif': 'image/tiff',
    'webp': 'image/webp',
    'ico': 'image/x-icon',
    'ppm': 'image/x-portable-pixmap',
    'pgm': 'image/x-portable-graymap',
    'pbm': 'image/x-portable-bitmap',
    'xpm': 'image/x-xpixmap',
    'xbm': 'image/x-xbitmap',

    # Vector formats
    'svg': 'image/svg+xml',
    'eps': 'application/postscript',
    'ai': 'application/postscript',
    'pdf': 'application/pdf',

    # RAW formats
    'raw': 'image/x-raw',
    'cr2': 'image/x-canon-cr2',
    'nef': 'image/x-nikon-nef',
    'arw': 'image/x-sony-arw',
    'dng': 'image/x-adobe-dng',
    'orf': 'image/x-olympus-orf',
    'rw2': 'image/x-panasonic-rw2',
    'pef': 'image/x-pentax-pef',
    'srw': 'image/x-samsung-srw',
    'raf': 'image/x-fuji-raf',

    # Professional formats
    'psd': 'image/vnd.adobe.photoshop',
    'psb': 'image/vnd.adobe.photoshop',
    'ai': 'application/postscript',
    'eps': 'application/postscript',
    'indd': 'application/x-indesign',

    # Legacy formats
    'pcx': 'image/x-pcx',
    'tga': 'image/x-tga',
    'sgi': 'image/x-sgi',
    'rgb': 'image/x-rgb',
    'rgba': 'image/x-rgba',
    'bw': 'image/x-bw',

    # 3D and model formats
    'obj': 'model/obj',
    'stl': 'model/stl',
    'ply': 'model/ply',
    'fbx': 'model/fbx',
    'dae': 'model/vnd.collada+xml',
    '3ds': 'model/3ds',
    'blend': 'application/x-blender',

    # Additional formats
    'heic': 'image/heic',
    'heif': 'image/heif',
    'avif': 'image/avif',
    'jxl': 'image/jxl',
    'jp2': 'image/jp2',
    'j2k': 'image/jp2',
    'jpx': 'image/jpx',
    'jpm': 'image/jpm',
    'mj2': 'image/mj2',
    'hdr': 'image/vnd.radiance',
    'exr': 'image/x-exr',
    'tga': 'image/x-tga',
    'sgi': 'image/x-sgi',
    'rgb': 'image/x-rgb',
    'rgba': 'image/x-rgba',
    'bw': 'image/x-bw',
    'pict': 'image/x-pict',
    'pct': 'image/x-pict',
    'mac': 'image/x-macpaint',
    'cut': 'image/x-cut',
    'dcm': 'application/dicom',
    'dicom': 'application/dicom'
}

# AI Model Provider Image Support Matrix
PROVIDER_IMAGE_SUPPORT = {
    "openai": {
        "supported_formats": ["png", "jpeg", "jpg", "gif", "webp"],
        "max_size_mb": 20,
        "max_dimensions": (2048, 2048),
        "preferred_format": "png"
    },
    "anthropic": {
        "supported_formats": ["png", "jpeg", "jpg", "gif", "webp"],
        "max_size_mb": 20,
        "max_dimensions": (2048, 2048),
        "preferred_format": "png"
    },
    "google": {
        "supported_formats": ["png", "jpeg", "jpg", "gif", "webp", "bmp"],
        "max_size_mb": 20,
        "max_dimensions": (2048, 2048),
        "preferred_format": "png"
    },
    "together": {
        "supported_formats": ["png", "jpeg", "jpg"],
        "max_size_mb": 10,
        "max_dimensions": (1024, 1024),
        "preferred_format": "png"
    },
    "ollama": {
        "supported_formats": ["png", "jpeg", "jpg", "gif", "webp"],
        "max_size_mb": 50,
        "max_dimensions": (4096, 4096),
        "preferred_format": "png"
    }
}


def get_image_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about an image file.

    Args:
        file_path: Path to the image file

    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(file_path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size": os.path.getsize(file_path),
                "mime_type": get_mime_type(file_path),
                "is_animated": getattr(img, 'is_animated', False),
                "n_frames": getattr(img, 'n_frames', 1)
            }

            # Get color depth
            if img.mode in ['RGB', 'RGBA']:
                info["color_depth"] = 24
            elif img.mode in ['L', 'LA']:
                info["color_depth"] = 8
            elif img.mode == 'P':
                info["color_depth"] = 8
            else:
                info["color_depth"] = "unknown"

            return info
    except Exception as e:
        logger.error(f"Failed to get image info for {file_path}: {e}")
        return {"error": str(e)}


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type for a file based on extension.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string
    """
    ext = Path(file_path).suffix.lower().lstrip('.')
    return IMAGE_MIME_TYPES.get(ext, 'application/octet-stream')


def is_supported_by_provider(file_path: str, provider: str) -> bool:
    """
    Check if an image format is supported by a specific AI provider.

    Args:
        file_path: Path to the image file
        provider: AI provider name

    Returns:
        True if supported, False otherwise
    """
    if provider not in PROVIDER_IMAGE_SUPPORT:
        return False

    ext = Path(file_path).suffix.lower().lstrip('.')
    supported_formats = PROVIDER_IMAGE_SUPPORT[provider]["supported_formats"]

    return ext in supported_formats


def get_optimal_conversion_target(file_path: str, provider: str) -> str:
    """
    Get the optimal target format for conversion based on provider.

    Args:
        file_path: Path to the source image
        provider: AI provider name

    Returns:
        Target format string
    """
    if provider not in PROVIDER_IMAGE_SUPPORT:
        return "png"  # Default fallback

    return PROVIDER_IMAGE_SUPPORT[provider]["preferred_format"]


def resize_image_if_needed(image_path: str, provider: str) -> str:
    """
    Resize image if it exceeds provider limits.

    Args:
        image_path: Path to the image
        provider: AI provider name

    Returns:
        Path to resized image (or original if no resize needed)
    """
    if provider not in PROVIDER_IMAGE_SUPPORT:
        return image_path

    limits = PROVIDER_IMAGE_SUPPORT[provider]
    max_width, max_height = limits["max_dimensions"]

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Check if resize is needed
            if width <= max_width and height <= max_height:
                return image_path

            # Calculate new dimensions maintaining aspect ratio
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)

            # Resize image
            resized_img = img.resize((new_width, new_height),
                                     Image.Resampling.LANCZOS)

            # Save to temporary file
            temp_path = tempfile.mktemp(suffix=f".{img.format.lower()}")
            resized_img.save(temp_path, format=img.format)

            logger.info(f"Resized image from {width}x{height} to " +
                        f"{new_width}x{new_height} for {provider}")

            return temp_path

    except Exception as e:
        logger.warning(f"Failed to resize image: {e}")
        return image_path


def convert_svg_to_png(svg_path: str,
                       output_path: Optional[str] = None
                       ) -> Optional[str]:
    """
    Convert SVG to PNG using cairosvg or other available converters.

    Args:
        svg_path: Path to SVG file
        output_path: Optional output path

    Returns:
        Path to converted PNG file or None if failed
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.png')

    try:
        # Try cairosvg first (most reliable)
        cairosvg.svg2png(url=svg_path, write_to=output_path)
        logger.info(f"Converted SVG to PNG using cairosvg: {output_path}")
        return output_path
    except Exception as e:
        logger.debug(f"cairosvg failed: {e}")

        # Try rsvg-convert (system command)
        try:
            result = subprocess.run(['rsvg-convert',
                                     '-f', 'png',
                                     '-o', output_path,
                                     svg_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=30)

            if result.returncode == 0:
                logger.info("Converted SVG to PNG using rsvg-convert: " +
                            f"{output_path}")
                return output_path
            else:
                logger.debug(f"rsvg-convert failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("rsvg-convert not available")

        # Try ImageMagick
        try:
            result = subprocess.run(['convert', svg_path, output_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=30)

            if result.returncode == 0:
                logger.info("Converted SVG to PNG using ImageMagick: " +
                            f"{output_path}")
                return output_path
            else:
                logger.debug(f"ImageMagick failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ImageMagick not available")

    logger.error("Failed to convert SVG to PNG - no suitable converter " +
                 "available")
    return None


def convert_vector_to_png(vector_path: str,
                          output_path: Optional[str] = None
                          ) -> Optional[str]:
    """
    Convert vector formats (SVG, EPS, AI, PDF) to PNG.

    Args:
        vector_path: Path to vector file
        output_path: Optional output path

    Returns:
        Path to converted PNG file or None if failed
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.png')

    mime_type = get_mime_type(vector_path)

    # Handle SVG files
    if mime_type == 'image/svg+xml':
        return convert_svg_to_png(vector_path, output_path)

    # Handle PostScript files (EPS, AI)
    elif mime_type == 'application/postscript':
        logger.info(f"Converting PostScript file to PNG: {vector_path}")
        try:
            # Try using ImageMagick for PostScript conversion
            result = subprocess.run(['convert', vector_path, output_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=60)

            if result.returncode == 0:
                logger.info("Converted PostScript to PNG using ImageMagick: " +
                            f"{output_path}")
                return output_path
            else:
                logger.debug(f"ImageMagick failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ImageMagick not available for PostScript " +
                           "conversion")

        # Try using Ghostscript for EPS files
        try:
            result = subprocess.run([
                'gs', '-sDEVICE=pngalpha', '-dNOPAUSE', '-dBATCH', '-dSAFER',
                f'-sOutputFile={output_path}', vector_path
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("Converted PostScript to PNG using Ghostscript: " +
                            f"{output_path}")
                return output_path
            else:
                logger.debug(f"Ghostscript failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Ghostscript not available for " +
                           "PostScript conversion")

    # Handle PDF files
    elif mime_type == 'application/pdf':
        logger.info(f"Converting PDF to PNG: {vector_path}")
        try:
            # Try using ImageMagick for PDF conversion
            result = subprocess.run(['convert',
                                     '-density',
                                     '150',
                                     vector_path,
                                     output_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=60)

            if result.returncode == 0:
                logger.info("Converted PDF to PNG using ImageMagick: " +
                            f"{output_path}")
                return output_path
            else:
                logger.debug(f"ImageMagick failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ImageMagick not available for PDF conversion")

        # Try using pdftoppm for PDF conversion
        try:
            result = subprocess.run([
                'pdftoppm', '-png', '-singlefile', vector_path,
                output_path.replace('.png', '')
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("Converted PDF to PNG using pdftoppm: " +
                            f"{output_path}")
                return output_path
            else:
                logger.debug(f"pdftoppm failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("pdftoppm not available for PDF conversion")

    logger.error("Failed to convert vector file to PNG - no suitable " +
                 f"converter available for {mime_type}")
    return None


def convert_to_standard_jpeg(input_path: str,
                             max_resolution: int = 512,
                             quality: int = 70
                             ) -> Optional[str]:
    """
    Convert any image to JPEG format with standardized settings.

    Args:
        input_path: Path to input image
        max_resolution: Maximum resolution for any side (default: 512px)
        quality: JPEG quality (1-100, default: 70)

    Returns:
        Path to converted JPEG file or None if failed
    """
    logger.info(f"Starting JPEG conversion for: {input_path}")

    # Check if this is a vector file
    mime_type = get_mime_type(input_path)
    vector_formats = ['image/svg+xml',
                      'application/postscript',
                      'application/pdf']

    if mime_type in vector_formats:
        logger.info(f"Detected vector file ({mime_type}), " +
                    "converting to PNG first")
        # Convert vector to PNG first
        png_path = convert_vector_to_png(input_path)
        if png_path:
            # Now convert PNG to JPEG
            jpeg_path = convert_to_standard_jpeg(png_path,
                                                 max_resolution,
                                                 quality)
            # Clean up the intermediate PNG file
            cleanup_temp_file(png_path)
            return jpeg_path
        else:
            logger.error("Failed to convert vector file to PNG: " +
                         f"{input_path}")
            return None

    try:
        with Image.open(input_path) as img:
            logger.debug(
                f"Original image format: {img.format}, " +
                f"mode: {img.mode}, size: {img.size}"
            )

            # Convert to RGB if needed (JPEG doesn't support transparency)
            if img.mode in ['RGBA', 'LA', 'P']:
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(
                        img,
                        mask=img.split()[-1] if img.mode == 'RGBA' else None
                )
                img = background
                logger.debug("Converted transparent image to RGB")
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                logger.debug(f"Converted image mode from {img.mode} to RGB")

            # Resize if needed while maintaining aspect ratio
            width, height = img.size
            if width > max_resolution or height > max_resolution:
                # Calculate new dimensions maintaining aspect ratio
                ratio = min(max_resolution / width, max_resolution / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                img = img.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to " +
                            f"{new_width}x{new_height}")
            else:
                logger.debug(f"Image size {width}x{height} is within limits")

            # Create output path
            output_path = tempfile.mktemp(suffix='.jpg')
            logger.debug(f"Output path: {output_path}")

            # Save as JPEG with specified quality
            img.save(output_path, 'JPEG', quality=quality, optimize=True)

            logger.info(
                f"Converted {input_path} to JPEG (quality: {quality}, " +
                f"max_res: {max_resolution}px): {output_path}"
            )
            return output_path

    except Exception as e:
        logger.error(f"Failed to convert {input_path} to JPEG: {e}")
        return None


def convert_image_format(input_path: str,
                         target_format: str = 'png',
                         quality: int = 95
                         ) -> Optional[str]:
    """
    Convert an image to a supported format.

    Args:
        input_path: Path to input image
        target_format: Target format (png, jpeg, webp, etc.)
        quality: Quality for lossy formats (1-100)

    Returns:
        Path to converted image or None if failed
    """
    try:
        with Image.open(input_path) as img:
            # Handle transparency for formats that don't support it
            if (target_format.lower() in ['jpeg',
                                          'jpg'] and img.mode in ['RGBA',
                                                                  'LA',
                                                                  'P']):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(
                        img,
                        mask=img.split()[-1] if img.mode == 'RGBA' else None
                )
                img = background
            elif img.mode == 'P':
                # Convert palette mode to RGBA
                img = img.convert('RGBA')

            # Create output path
            output_path = tempfile.mktemp(suffix=f'.{target_format}')

            # Save with appropriate format and options
            if target_format.lower() == 'png':
                img.save(output_path, 'PNG')
            elif target_format.lower() in ['jpeg', 'jpg']:
                img.save(output_path, 'JPEG', quality=quality)
            elif target_format.lower() == 'webp':
                img.save(output_path, 'WEBP', quality=quality)
            elif target_format.lower() == 'gif':
                img.save(output_path, 'GIF')
            else:
                img.save(output_path, target_format.upper())

            logger.info(
                f"Converted {input_path} to {target_format}: {output_path}"
            )
            return output_path

    except Exception as e:
        logger.error(f"Failed to convert image {input_path} to " +
                     f"{target_format}: {e}")
        return None


def optimize_image_for_provider(image_path: str, provider: str) -> str:
    """
    Optimize image for a specific AI provider.

    Args:
        image_path: Path to the image
        provider: AI provider name

    Returns:
        Path to optimized image
    """
    if provider not in PROVIDER_IMAGE_SUPPORT:
        return image_path

    # Use standard JPEG conversion for all images
    converted_path = convert_to_standard_jpeg(
        image_path,
        max_resolution=512,
        quality=70
    )

    if converted_path:
        logger.info(
            f"Optimized image for {provider} using standard JPEG conversion"
            )
        return converted_path

    # Fallback to original provider-specific optimization if
    # standard conversion fails
    logger.warning(
        "Standard JPEG conversion failed, using provider-specific optimization"
    )

    # Check if format is already supported
    if is_supported_by_provider(image_path, provider):
        # Just resize if needed
        return resize_image_if_needed(image_path, provider)

    # Convert to supported format
    target_format = get_optimal_conversion_target(image_path, provider)

    # Handle SVG specifically
    if get_mime_type(image_path) == 'image/svg+xml':
        converted_path = convert_svg_to_png(image_path)
        if converted_path:
            return resize_image_if_needed(converted_path, provider)

    # Convert other formats
    converted_path = convert_image_format(image_path, target_format)
    if converted_path:
        return resize_image_if_needed(converted_path, provider)

    # Fallback to original
    logger.warning(f"Failed to optimize image for {provider}, using original")
    return image_path


def get_supported_conversion(file_path: str) -> Optional[Tuple[callable, str]]:
    """
    Get supported conversion function and target format for a file.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (conversion_function, target_format) or None
    """
    mime_type = get_mime_type(file_path)

    if mime_type == 'image/svg+xml':
        return convert_svg_to_png, 'png'
    elif mime_type in ['image/tiff', 'image/tif']:
        return lambda x, y='png': convert_image_format(x, y), 'png'
    elif mime_type in ['image/bmp', 'image/x-bmp']:
        return lambda x, y='png': convert_image_format(x, y), 'png'
    elif mime_type in ['image/heic', 'image/heif']:
        return lambda x, y='jpeg': convert_image_format(x, y), 'jpeg'
    elif mime_type in ['image/avif']:
        return lambda x, y='png': convert_image_format(x, y), 'png'
    elif mime_type in ['image/jxl']:
        return lambda x, y='png': convert_image_format(x, y), 'png'

    return None


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file safely.

    Args:
        file_path: Path to temporary file
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


def batch_optimize_images(image_paths: List[str], provider: str) -> List[str]:
    """
    Optimize multiple images for a provider.

    Args:
        image_paths: List of image paths
        provider: AI provider name

    Returns:
        List of optimized image paths
    """
    optimized_paths = []

    for image_path in image_paths:
        try:
            optimized_path = optimize_image_for_provider(image_path, provider)
            optimized_paths.append(optimized_path)
            logger.debug(f"Optimized {image_path} for {provider}: " +
                         f"{optimized_path}")
        except Exception as e:
            logger.warning(f"Failed to optimize {image_path} for {provider}:" +
                           f" {e}")
            optimized_paths.append(image_path)  # Use original as fallback

    return optimized_paths


def validate_image_for_provider(image_path: str,
                                provider: str
                                ) -> Dict[str, Any]:
    """
    Validate if an image meets provider requirements.

    Args:
        image_path: Path to the image
        provider: AI provider name

    Returns:
        Validation result dictionary
    """
    if provider not in PROVIDER_IMAGE_SUPPORT:
        return {"valid": False, "error": f"Unknown provider: {provider}"}

    provider_config = PROVIDER_IMAGE_SUPPORT[provider]

    try:
        image_info = get_image_info(image_path)
        if "error" in image_info:
            return {"valid": False, "error": image_info["error"]}

        # Check format support
        format_supported = is_supported_by_provider(image_path, provider)

        # Check size limits
        file_size_mb = image_info["file_size"] / (1024 * 1024)
        size_ok = file_size_mb <= provider_config["max_size_mb"]

        # Check dimension limits
        width, height = image_info["size"]
        max_width, max_height = provider_config["max_dimensions"]
        dimensions_ok = width <= max_width and height <= max_height

        valid = format_supported and size_ok and dimensions_ok

        return {
            "valid": valid,
            "format_supported": format_supported,
            "size_ok": size_ok,
            "dimensions_ok": dimensions_ok,
            "file_size_mb": file_size_mb,
            "dimensions": image_info["size"],
            "format": image_info["format"],
            "recommendations": []
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


def batch_convert_to_standard_jpeg(image_paths: List[str],
                                   max_resolution: int = 512,
                                   quality: int = 70
                                   ) -> List[str]:
    """
    Convert multiple images to standard JPEG format.

    Args:
        image_paths: List of image paths to convert
        max_resolution: Maximum resolution for any side (default: 512px)
        quality: JPEG quality (1-100, default: 70)

    Returns:
        List of converted JPEG file paths (or original paths if
        conversion failed)
    """
    converted_paths = []

    for image_path in image_paths:
        try:
            converted_path = convert_to_standard_jpeg(
                image_path,
                max_resolution=max_resolution,
                quality=quality
            )

            if converted_path:
                converted_paths.append(converted_path)
                logger.debug(
                    f"Successfully converted {image_path} to JPEG format"
                )
            else:
                converted_paths.append(image_path)  # Use original as fallback
                logger.warning(
                    f"Failed to convert {image_path} to JPEG, using original"
                )

        except Exception as e:
            converted_paths.append(image_path)  # Use original as fallback
            logger.warning(
                f"Error converting {image_path} to JPEG: {e}, using original"
            )

    return converted_paths
