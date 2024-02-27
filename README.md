

# Ball Detection and Counting

This Python script utilizes the OpenCV library to process images, detect circles, and identify colored balls within those circles. It then counts the number of scratched and unscratched balls and outputs an annotated image with this information.

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy


The script performs the following main steps:

1. **Argument Parsing**: Parses command-line arguments to accept input and output image paths.
   
2. **Image Loading**: Loads the input image and resizes it to predefined dimensions.

3. **Preprocessing**: Converts the image to grayscale and removes the green background.

4. **Circle Detection**: Utilizes the Hough Circle Transform to detect circles in the preprocessed image.

5. **Color Identification**: Identifies balls of different colors using predefined color masks.

6. **Processing Circles and Masks**: Iterates over the detected circles and color masks to determine the presence and color of balls within each circle.

7. **Output Generation**: Produces an annotated output image with text indicating the number of scratched and unscratched balls.

## Examples

Input Image             |  Output Image
:-------------------------:|:-------------------------:
![Input Image](input_image.jpg)  |  ![Output Image](output_image.jpg)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

