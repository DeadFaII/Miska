from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops
import numpy as np
from scipy.ndimage import convolve
from scipy.fftpack import dst, idst

# Define Sobel operators for horizontal and vertical directions
sobel_horizontal = (-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1)

sobel_vertical = (1, 2, 1,
                  0, 0, 0,
                  -1, -2, -1)


# Specular/Gloss => Metallic/Roughness
def albedo_sg2mr(diffuse, specular, metalness):
    """
    Convert a Specular/Glossy diffuse image to Metallic/Roughness albedo.

    Parameters:
    - diffuse (PIL.Image): The original diffuse image in Specular/Glossy format.
    - specular (PIL.Image): The specular map.
    - metalness (PIL.Image): The metalness map.

    Returns:
    - PIL.Image: The converted diffuse image in Metallic/Roughness format.
    """
    return Image.composite(diffuse, specular, ImageOps.invert(metalness))


def roughness_sg2mr(glossiness):
    """
    Convert a glossiness map to a roughness map.

    Parameters:
    - glossiness (PIL.Image): The original glossiness map.

    Returns:
    - PIL.Image: The converted roughness map.
    """
    return ImageOps.invert(glossiness)


# Metallic/Roughness => Specular/Gloss
def diffuse_mr2sg(albedo, metalness):
    """
    Convert a Metallic/Roughness albedo image to Specular/Glossy diffuse.

    Parameters:
    - albedo (PIL.Image): The original albedo image in Metallic/Roughness format.
    - metalness (PIL.Image): The metalness map.

    Returns:
    - PIL.Image: The converted albedo image in Specular/Glossy format.
    """
    fill_layer = Image.new("RGB", albedo.size, "#000000")
    return Image.composite(albedo, fill_layer, ImageOps.invert(metalness))


def specular_mr2sg(albedo, metalness):
    """
    Convert a metallic roughness specular image to Specular/Glossy.

    Parameters:
    - albedo (PIL.Image): The original albedo image in Metallic/Roughness format.
    - metalness (PIL.Image): The metalness map.

    Returns:
    - PIL.Image: The converted specular image in Specular/Glossy format.
    """
    fill_layer = Image.new("RGB", albedo.size, "#383838")
    return Image.composite(albedo, fill_layer, metalness)


def glossiness_mr2sg(roughness):
    """
    Convert a roughness map to a glossiness map.

    Parameters:
    - roughness (PIL.Image): The original roughness map.

    Returns:
    - PIL.Image: The converted glossiness map.
    """
    return ImageOps.invert(roughness)


def height2normal(height, api="DirectX"):
    """
    Convert a height map to a normal map.

    Parameters:
    - height (PIL.Image): The height map.
    - api (str): The API format for the normal map. Can be either "DirectX" or "OpenGL".

    Returns:
    - PIL.Image: The generated normal map.
    """

    def creating_channel(sobel_direction):
        """
        Create a normal map channel using the Sobel operator.

        Parameters:
        - sobel_direction (str): The direction of the Sobel operator ("h" for horizontal, "v" for vertical).

        Returns:
        - PIL.Image: The normal map channel image.
        """
        if sobel_direction == 'h':
            sobel_filter = height.filter(ImageFilter.Kernel((3, 3), sobel_horizontal, 1, 0))
            flipped_sobel_filter = height.filter(
                ImageFilter.Kernel((3, 3), tuple(-i for i in sobel_horizontal), 1, 0))
        elif sobel_direction == 'v':
            sobel_filter = height.filter(ImageFilter.Kernel((3, 3), sobel_vertical, 1, 0))
            flipped_sobel_filter = height.filter(ImageFilter.Kernel((3, 3), tuple(-i for i in sobel_vertical), 1, 0))

        background = Image.new('L', height.size, color=128)  # Create a layer with gray fill

        layer1 = Image.new('L', height.size, color=0)  # Create a layer with black fill
        layer2 = Image.new('L', height.size, color=255)  # Create a layer with white fill

        # Combine layers
        image_temp = Image.composite(layer1, background, (sobel_filter).convert('L'))
        image_result = Image.composite(layer2, image_temp, (flipped_sobel_filter).convert('L'))

        return ImageEnhance.Contrast(image_result).enhance(0.7)  # Reduce contrast

    r = creating_channel('h')
    g = creating_channel('v')
    b = Image.new('L', height.size, color=255)

    if api == "OpenGL":
        g = ImageOps.invert(g)

    return Image.merge("RGB", (r, g, b))


def normal2height(normal, api="DirectX"):
    """
    Convert a normal map to a height map.

    Parameters:
    - normal (PIL.Image): The normal map.
    - api (str): The API format for the normal map. Can be either "DirectX" or "OpenGL".

    Returns:
    - PIL.Image: The generated height map.
    """
    # Convert the normal map to a numpy array and normalize values to -1 to 1
    normal_array = ((np.asarray(normal) / 255.0) - 0.5) * 2.0

    if api == "OpenGL":
        normal_array[:, :, 1] = -normal_array[:, :, 1]

    # Pad the array by 100 pixels
    padded_normal_array = np.pad(normal_array, ((100, 100), (100, 100), (0, 0)), mode="wrap")

    # Extract gradients from the normal map
    gradient_x = padded_normal_array[:, :, 0]
    gradient_y = padded_normal_array[:, :, 1]

    # Initialize the Laplacian array
    height, width = gradient_x.shape
    laplacian = np.zeros((height, width))

    # Compute the Laplacian by calculating the second derivatives using gradients
    laplacian[1:, :] += gradient_y[1:, :] - gradient_y[:-1, :]
    laplacian[:, 1:] += gradient_x[:, 1:] - gradient_x[:, :-1]

    # Solve Poisson's equation to get the height map
    # Apply DST to rows
    dst_laplacian = dst(laplacian, type=1, axis=0)
    # Apply DST to columns
    dst_laplacian = dst(dst_laplacian, type=1, axis=1)

    # Divide by eigenvalues
    M, N = laplacian.shape
    eigenvalues = (2 * np.cos(np.pi * np.arange(1, M + 1) / (M + 1))[:, None] - 2) + \
                  (2 * np.cos(np.pi * np.arange(1, N + 1) / (N + 1)) - 2)
    dst_laplacian /= eigenvalues

    # Apply inverse DST to columns
    inverse_dst = idst(dst_laplacian, type=1, axis=1)
    # Apply inverse DST to rows
    height_map = idst(inverse_dst, type=1, axis=0)

    # Normalize
    height_map /= (2 * (M + 1)) * (2 * (N + 1))

    # Normalize the height map to the range 0-255
    normalized_height_map = (
                ((height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))) * 255).astype(np.uint8)

    # Crop the padded area and invert the image
    cropped_height_map = ImageOps.crop(Image.fromarray(normalized_height_map), (100, 100, 100, 100))
    inverted_height_map = ImageOps.invert(cropped_height_map)

    return ImageEnhance.Contrast(inverted_height_map).enhance(1.35)  # Enhance the contrast


def normal2curvature(normal, api="DirectX"):
    """
    Convert a normal map to a curvature map.

    Parameters:
    - normal (PIL.Image): The normal map.
    - api (str): The API format for the normal map. Can be either "DirectX" or "OpenGL".

    Returns:
    - PIL.Image: The generated curvature map.
    """
    w, h = normal.size

    # Normalize the normal map to the range [-1, 1]
    normalize = np.array(normal).astype(np.float32) / 255.0 * 2.0 - 1.0

    if api == "OpenGL":
        normalize[:, :, 1] = -normalize[:, :, 1]

    base_map_array = np.zeros((h, w), dtype=np.float32)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]]) / 8.0

    # Calculate the x and y direction differences for the normal map
    nx_diff = (np.roll(normalize[:, :, 0], -1, axis=1) - np.roll(normalize[:, :, 0], 1, axis=1)) / 2.0
    ny_diff = (np.roll(normalize[:, :, 1], -1, axis=0) - np.roll(normalize[:, :, 1], 1, axis=0)) / 2.0

    # Refine the base curvature map
    for _ in range(10):
        average_neighbors = convolve(base_map_array, kernel, mode="wrap")

        # Update the map using the average of neighbors and differences from normal maps
        base_map_array = average_neighbors + (nx_diff + ny_diff) / 2.0

    # Create the base map from the array
    base_map = (((base_map_array - np.min(base_map_array)) / np.max(
        base_map_array - np.min(base_map_array))) * 255).astype(np.uint8)

    sobel_x = normal.getchannel("R").filter(ImageFilter.Kernel((3, 3), sobel_horizontal, 1, 0))
    sobel_y = normal.getchannel("G").filter(ImageFilter.Kernel((3, 3), sobel_vertical, 1, 0))

    # Normalize to [0, 1]
    sobel_x_data = np.array(sobel_x).astype(np.float32) / 255.0
    sobel_y_data = np.array(sobel_y).astype(np.float32) / 255.0

    # Sobel operator to detect edges
    sobel_operator = np.sqrt(sobel_x_data ** 2 + sobel_y_data ** 2)

    sobel_operator_norm = sobel_operator / np.max(sobel_operator)

    detected_edges = (sobel_operator_norm * 255).astype(np.uint8)

    curvature = Image.blend(Image.fromarray(base_map), Image.fromarray(detected_edges), 0.3)

    return ImageEnhance.Brightness(curvature).enhance(1.3)


def height2ao(height):
    """
    Convert a height map to an ambient occlusion (AO) map.

    Parameters:
    - height (PIL.Image): The height map.

    Returns:
    - PIL.Image: The generated AO map.
    """
    # Convert the height map to a numpy array and normalize to the range [0, 1]
    height_array = np.asarray(height, dtype=np.float32) / 255.0
    height_array = np.pad(height_array, ((10, 10), (10, 10)), mode="wrap")  # Add padding of 10px
    ao_array = np.ones_like(height_array)  # Initialize the AO array with ones

    # Get the dimensions of the height map
    w, h = height_array.shape

    # Define offsets for a 5x5 neighborhood
    offsets = [(di, dj) for di in range(-2, 3) for dj in range(-2, 3) if not (di == 0 and dj == 0)]

    # Iterate over each pixel in the height map
    for i in range(2, w - 2):
        for j in range(2, h - 2):
            # Get the height of the current pixel
            current_height = height_array[i, j]

            # Check all 24 neighboring pixels (5x5)
            for di, dj in offsets:
                neighbor_height = height_array[i + di, j + dj]
                distance = max(abs(di), abs(dj))
                if neighbor_height > current_height:
                    # If the height of the neighboring pixel is greater, reduce the AO value of the current pixel
                    ao_array[i, j] -= 2 * (neighbor_height - current_height) / (2 ** (distance - 1))

    # Normalize the AO values to the range [0, 1]
    ao_array = np.clip(ao_array, 0, 1)

    # Convert the AO array to an image and crop the padding
    ao_image = Image.fromarray((ao_array * 255).astype(np.uint8))
    ao_image = ImageOps.crop(ao_image, (10, 10, 10, 10))

    # Apply the "Hard Light" blending effect to the image
    ao_image = ImageChops.hard_light(ao_image, height)

    # Apply Gaussian Blur filter and return the image
    return ao_image.filter(ImageFilter.GaussianBlur(1.2))