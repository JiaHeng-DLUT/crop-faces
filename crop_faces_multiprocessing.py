import cv2
import math
import os
from mtcnn import MTCNN
from multiprocessing import Pool

base_dir = os.path.dirname(os.path.abspath(__file__))
detector = MTCNN()
(cropped_height, cropped_width) = (160, 160)

def crop_image(image_abs_path):
    image = cv2.read(image_abs_path)
    (image_width, image_height, image_channels) = image.shape
    if image_width < cropped_width or image_height < cropped_height:
        print("Resolution too low! Pass current frame.")
        return
    result = detector.detect_faces(image)
    # Crop all faces
    print("%d faces detected." % (len(result)))
    if result:
        for i in range(len(result)):
            print("Face %d: " % (i), end='')
            bounding_box = result[i]['box']
            # keypoints = result[i]['keypoints']
            # color = (255,0,0)
            # cv2.rectangle(image,
            #               (bounding_box[0], bounding_box[1]),
            #               (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            #               color,
            #               2)
            # cv2.circle(image, (keypoints['left_eye']), 2, color, 2)
            # cv2.circle(image, (keypoints['right_eye']), 2, color, 2)
            # cv2.circle(image, (keypoints['nose']), 2, color, 2)
            # cv2.circle(image, (keypoints['mouth_left']), 2, color, 2)
            # cv2.circle(image, (keypoints['mouth_right']), 2, color, 2)
            if bounding_box[2] > cropped_width or bounding_box[3] > cropped_height:
                print("Too large! Pass current face.")
                continue
            center_x = int(bounding_box[0] + bounding_box[2] / 2)
            center_y = int(bounding_box[1] + bounding_box[3] / 2)
            print("(%d, %d)" % (center_x, center_y))
            dx = cropped_width / 2
            dy = cropped_height / 2
            (left, right) = (center_x - math.floor(dx), center_x + math.ceil(dx))
            (top, down) = (center_y - math.floor(dy), center_y + math.ceil(dy))
            if left < 0:
                (left, right) = (0, cropped_width)
            if top < 0:
                (top, down) = (0, cropped_height)
            if right > image_width:
                left = image_width - cropped_width
                right = image_width
            if down > image_height:
                top = image_height - cropped_height
                down = image_height
            # print("Cropped erea:")
            # print("Top-left corner: (%d, %d)" % (left, top))
            # print("Down-right corner: (%d, %d)" % (right, down))
            face = image[top : down, left : right]
            face_dir = os.path.dirname(image_abs_path).replace("YTFaces", "YTFaces_HR")
            if not os.path.exists(face_dir):
                os.makedirs(face_dir)
            face_abs_path = image_abs_path.replace("YTFaces", "YTFaces_HR") + "_face%d.jpg" % (i)
            cv2.imwrite(face_abs_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    g = os.walk(os.path.join(base_dir, r"YTFaces"))
    for (path, dir_list, file_list) in g:
        images_abs_path = [os.path.join(path, file_name) for file_name in file_list if file_name.endswith(".jpg")]
        with Pool(100) as p:
            p.map(crop_image, images_abs_path)