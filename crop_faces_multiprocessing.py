import cv2
import math
import os
import time
from mtcnn import MTCNN
from multiprocessing import Pool

detector = MTCNN()

def crop_face(image_path, cropped_height=224, cropped_width=224):
    print(image_path + ': ', end='')
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    (image_width, image_height, image_channels) = image.shape
    if image_width < cropped_width or image_height < cropped_height:
        print("Resolution too low! Pass current frame.")
        return
    result = detector.detect_faces(image)
    # Crop all faces
    print("%d faces detected." % (len(result)))
    if result:
        for i in range(len(result)):
            # print("Face %d: " % (i), end='')
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
            # print("(%d, %d)" % (center_x, center_y))
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
            face_dir = os.path.dirname(image_path).replace("YTFaces", "YTFaces_HR_" + str(cropped_width) + "X" + str(cropped_height))
            if not os.path.exists(face_dir):
                os.makedirs(face_dir)
            face_abs_path = image_path.replace("YTFaces", "YTFaces_HR_" + str(cropped_width) + "X" + str(cropped_height)) + "_face%d.jpg" % (i)
            cv2.imwrite(face_abs_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    return

if __name__ == '__main__':
    p = Pool(processes = 4)
    start_time = time.time()
    cnt = 0
    for (path, dir_list, file_list) in os.walk(r'.\YTFaces'):
        print(path)
        images_path_list = [os.path.join(path, file_name) for file_name in file_list if file_name.endswith(".jpg")]
        for image_path in images_path_list:
            cnt += 1
            p.apply_async(crop_face(image_path))
    p.close()
    p.join()
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    print(cnt)
