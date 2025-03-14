import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\samee\OneDrive\Pictures\Screenshots\estimage opencv.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [ (0,0.87*height),(681,358),(width,0.87*height),(width,height),(0,height)]
def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img,lines):
    imgc = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=3)
    img = cv2.addWeighted(img,0.8,blank_image,1,0.0)
    return img


gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_edge = cv2.Canny(gray_image,250,250)
cropped_image = region_of_interest(canny_edge,np.array([region_of_interest_vertices],np.int32),)
#image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
# plt.imshow(cropped_image)
# plt.show()
lines = cv2.HoughLinesP(cropped_image,rho = 6,
                        theta=np.pi/60,
                        threshold=96,
                        lines=np.array([]),
                        minLineLength=45,
                        maxLineGap=55)
image_with_lines = draw_the_lines(image,lines)
cv2.imshow("hi",image_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
# if image_with_lines is None:
#     print("Error: Image not found. Check the file path.")
# else:
#     image_rgb = cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB)
#     plt.imshow(image_with_lines)
#     plt.axis("off")  # Hide axis for better visualization
#     plt.show()
