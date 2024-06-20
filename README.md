# Miska-Python-Module-
Module for converting texture maps.

### README.md

# Texture Map Conversion Module

I wrote this module for a university thesis, it may contain bugs, but maybe it will help someone in writing a plugin or module) It provides functions to convert between different types of texture maps commonly used in 3D graphics. It includes functions to convert between Specular/Gloss and Metallic/Roughness formats, as well as functions to create normal maps, height maps, curvature maps, and ambient occlusion maps from various input maps.

*To learn more about converting between pipelines, I recommend reading this article: [PBR Texture Conversion](https://marmoset.co/posts/pbr-texture-conversion/)*

## Functions

### 1. `albedo_sg2mr`
Convert a Specular/Glossy diffuse image to Metallic/Roughness albedo.

```python
def albedo_sg2mr(diffuse, specular, metalness):
    return Image.composite(diffuse, specular, ImageOps.invert(metalness))
```

**Process**:
- Takes a diffuse texture (`diffuse`).
- Takes a specular texture (`specular`).
- Creates a mask by inverting the metalness texture (`metalness`).
- Blends the diffuse and specular textures based on the inverted metalness mask.

### 2. `roughness_sg2mr`
Convert a glossiness map to a roughness map.

```python
def roughness_sg2mr(glossiness):
    return ImageOps.invert(glossiness)
```

**Process**:
- Inverts the colors of the glossiness texture. Dark areas become light and vice versa.

### 3. `diffuse_mr2sg`
Convert a Metallic/Roughness albedo image to Specular/Glossy diffuse.

```python
def diffuse_mr2sg(albedo, metalness):
    fill_layer = Image.new("RGB", albedo.size, "#000000")
    return Image.composite(albedo, fill_layer, ImageOps.invert(metalness))
```

**Process**:
- Creates a black fill layer (`fill_layer`).
- Inverts the metalness map (`metalness`).
- Blends the albedo texture with the black fill layer based on the inverted metalness mask.

### 4. `specular_mr2sg`
Convert a metallic roughness specular image to Specular/Glossy.

```python
def specular_mr2sg(albedo, metalness):
    fill_layer = Image.new("RGB", albedo.size, "#383838")
    return Image.composite(albedo, fill_layer, metalness)
```

**Process**:
- Creates a dark gray fill layer (`fill_layer`).
- Blends the albedo texture with the dark gray fill layer based on the metalness mask.

### 5. `glossiness_mr2sg`
Convert a roughness map to a glossiness map.

```python
def glossiness_mr2sg(roughness):
    return ImageOps.invert(roughness)
```

**Process**:
- Inverts the colors of the roughness texture. Dark areas become light and vice versa.

### 6. `height2normal`
Convert a height map to a normal map.

```python
def height2normal(height, api="DirectX"):
    def creating_channel(sobel_direction):
        if sobel_direction == 'h':
            sobel_filter = height.filter(ImageFilter.Kernel((3, 3), sobel_horizontal, 1, 0))
            flipped_sobel_filter = height.filter(
                ImageFilter.Kernel((3, 3), tuple(-i for i in sobel_horizontal), 1, 0))
        elif sobel_direction == 'v':
            sobel_filter = height.filter(ImageFilter.Kernel((3, 3), sobel_vertical, 1, 0))
            flipped_sobel_filter = height.filter(ImageFilter.Kernel((3, 3), tuple(-i for i in sobel_vertical), 1, 0))

        background = Image.new('L', height.size, color=128)
        layer1 = Image.new('L', height.size, color=0)
        layer2 = Image.new('L', height.size, color=255)

        image_temp = Image.composite(layer1, background, (sobel_filter).convert('L'))
        image_result = Image.composite(layer2, image_temp, (flipped_sobel_filter).convert('L'))

        return ImageEnhance.Contrast(image_result).enhance(0.7)

    r = creating_channel('h')
    g = creating_channel('v')
    b = Image.new('L', height.size, color=255)

    if api == "OpenGL":
        g = ImageOps.invert(g)

    return Image.merge("RGB", (r, g, b))
```

**Process**:
- Creates horizontal and vertical channels using the Sobel operator.
- The `r` channel is horizontal, `g` is vertical, and `b` is filled with white color.
- For "OpenGL" format, the `g` channel is inverted.
- Combines the three channels into a normal map.

### 7. `normal2height`
Convert a normal map to a height map.

```python
def normal2height(normal, api="DirectX"):
    normal_array = ((np.asarray(normal) / 255.0) - 0.5) * 2.0

    if api == "OpenGL":
        normal_array[:, :, 1] = -normal_array[:, :, 1]

    padded_normal_array = np.pad(normal_array, ((100, 100), (100, 100), (0, 0)), mode="wrap")

    gradient_x = padded_normal_array[:, :, 0]
    gradient_y = padded_normal_array[:, :, 1]

    height, width = gradient_x.shape
    laplacian = np.zeros((height, width))

    laplacian[1:, :] += gradient_y[1:, :] - gradient_y[:-1, :]
    laplacian[:, 1:] += gradient_x[:, 1:] - gradient_x[:, :-1]

    dst_laplacian = dst(laplacian, type=1, axis=0)
    dst_laplacian = dst(dst_laplacian, type=1, axis=1)

    M, N = laplacian.shape
    eigenvalues = (2 * np.cos(np.pi * np.arange(1, M + 1) / (M + 1))[:, None] - 2) + \
                  (2 * np.cos(np.pi * np.arange(1, N + 1) / (N + 1)) - 2)
    dst_laplacian /= eigenvalues

    inverse_dst = idst(dst_laplacian, type=1, axis=1)
    height_map = idst(inverse_dst, type=1, axis=0)

    height_map /= (2 * (M + 1)) * (2 * (N + 1))

    normalized_height_map = (
                ((height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))) * 255).astype(np.uint8)

    cropped_height_map = ImageOps.crop(Image.fromarray(normalized_height_map), (100, 100, 100, 100))
    inverted_height_map = ImageOps.invert(cropped_height_map)

    return ImageEnhance.Contrast(inverted_height_map).enhance(1.35)
```

**Process**:
- Converts the normal map to a numpy array and normalizes values to the range -1 to 1.
- Adds padding of 100 pixels.
- Extracts gradients from the normal map.
- Computes the Laplacian using the gradients.
- Solves Poisson's equation to get the height map using DST (Discrete Sine Transform).
- Normalizes the height map to the range 0-255.
- Crops the padding and inverts the image.
- Enhances the contrast.

### 8. `normal2curvature`
Convert a normal map to a curvature map.

```python
def normal2curvature(normal, api="DirectX"):
    w, h = normal.size

    normalize = np.array(normal).astype(np.float32) / 255.0 * 2.0 - 1.0

    if api == "OpenGL":
        normalize[:, :, 1] = -normalize[:, :, 1]

    base_map_array = np.zeros((h, w), dtype=np.float32)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]]) / 8.0

    nx_diff = (np.roll(normalize[:, :, 0], -1, axis=1) - np.roll(normalize[:, :, 0], 1, axis=1)) / 2.0
    ny_diff = (np.roll(normalize[:, :, 1], -1, axis=0) - np.roll(normalize[:, :, 1], 1, axis=0)) / 2.0

    for _ in range(10):
        average_neighbors = convolve(base_map_array, kernel, mode="wrap")
        base_map_array = average_neighbors + (nx_diff + ny_diff) / 2.0

    base_map = (((base_map_array - np.min(base_map_array)) / np.max(
        base_map_array - np.min(base_map_array))) * 255).astype(np.uint8)

    sobel_x = normal.getchannel("R").filter(ImageFilter.Kernel((3, 3), sobel_horizontal, 1, 0))
    sobel_y = normal.getchannel("G").filter(ImageFilter.Kernel((3, 3), sobel_vertical, 1, 0))

    sobel_x_data = np.array(sobel_x).astype(np.float32) / 255.0
    sobel_y_data = np.array(sobel_y).astype(np.float32) / 255.0

    sob

el_operator = np.sqrt(sobel_x_data ** 2 + sobel_y_data ** 2)

    sobel_operator_norm = sobel_operator / np.max(sobel_operator)

    detected_edges = (sobel_operator_norm * 255).astype(np.uint8)

    curvature = Image.blend(Image.fromarray(base_map), Image.fromarray(detected_edges), 0.3)

    return ImageEnhance.Brightness(curvature).enhance(1.3)
```

**Process**:
- Normalizes the normal map to the range -1 to 1.
- Initializes the base curvature map.
- Computes the differences in the x and y directions.
- Refines the base curvature map using convolution.
- Detects edges using the Sobel operator.
- Blends the base curvature map with detected edges.
- Enhances the brightness for better visualization.

### 9. `height2ao`
Convert a height map to an ambient occlusion (AO) map.

```python
def height2ao(height):
    height_array = np.asarray(height, dtype=np.float32) / 255.0
    height_array = np.pad(height_array, ((10, 10), (10, 10)), mode="wrap")
    ao_array = np.ones_like(height_array)

    w, h = height_array.shape

    offsets = [(di, dj) for di in range(-2, 3) for dj in range(-2, 3) if not (di == 0 and dj == 0)]

    for i in range(2, w - 2):
        for j in range(2, h - 2):
            current_height = height_array[i, j]
            for di, dj in offsets:
                neighbor_height = height_array[i + di, j + dj]
                distance = max(abs(di), abs(dj))
                if neighbor_height > current_height:
                    ao_array[i, j] -= 2 * (neighbor_height - current_height) / (2 ** (distance - 1))

    ao_array = np.clip(ao_array, 0, 1)

    ao_image = Image.fromarray((ao_array * 255).astype(np.uint8))
    ao_image = ImageOps.crop(ao_image, (10, 10, 10, 10))

    ao_image = ImageChops.hard_light(ao_image, height)

    return ao_image.filter(ImageFilter.GaussianBlur(1.2))
```

**Process**:
- Converts the height map to a numpy array and normalizes to the range 0-1.
- Adds padding of 10 pixels.
- Initializes the ambient occlusion (AO) array.
- Calculates the AO value based on the height difference with the neighboring pixels, in simple words, checks if any of its neighbors are taller than the given pixel and if so, dims it based on the height difference.
- Normalizes the AO values to the range 0-1.
- Converts the AO array to an image and crops the padding.
- Applies the "Hard Light" blending effect.
- Applies Gaussian Blur for smoothing.

## Dependencies

- Python 3.x
- PIL (Pillow)
- NumPy
- SciPy

## Installation

To install the required packages, run:

```bash
pip install pillow numpy scipy
```

## Usage

To use any of the functions, simply import them and pass the appropriate images as arguments. For example:

```python
from PIL import Image
from miska import albedo_sg2mr

diffuse = Image.open("diffuse.png")
specular = Image.open("specular.png")
metalness = Image.open("metalness.png")

converted_image = albedo_sg2mr(diffuse, specular, metalness)
converted_image.save("converted_image.png")
```
