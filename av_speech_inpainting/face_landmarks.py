from glob import glob
import os
import numpy as np
import cv2
import dlib
from imutils import face_utils


FACIAL_LANDMARKS_IDXS = dict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])

# Landmark no. 33 is the location of nose tip
def adjust_landmarks(landmarks, anchor_landmark=33):
    #landmarks = landmarks.reshape((-1, 68, 2))
    # Subtract x-y coordinates of the anchor landmark
    adjusted_landmarks = landmarks - np.expand_dims(landmarks[:, anchor_landmark], axis=1)
    # Ids of the anchor landmarks to be removed from feature vector
    deleted_ids = list(range(anchor_landmark * 2, landmarks.size, 136)) + list(range(anchor_landmark * 2 + 1, landmarks.size, 136))

    return np.delete(adjusted_landmarks, deleted_ids)


def get_motion_vector(landmarks, delta=1, anchor_landmark=-1):
    if anchor_landmark >= 0:
        features = adjust_landmarks(landmarks, anchor_landmark)
    if delta > 0:
        features = np.zeros_like(landmarks)
        features[1:] = landmarks[1:] - landmarks[:-1]
        if delta == 2:
            features = features[1:] - features[:-1]

    return features


def extract_face_landmarks(video_filename, predictor_params, refresh_size=8):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_params)
    tracker = dlib.correlation_tracker()

    cap = cv2.VideoCapture(video_filename)

    tracking_face = False # Keep track if we are using tracker
    i = 0 # Number of frames without detection
    landmarks = []
    face_rects = []
    rect = None
    
    while cap.isOpened():
        ret, frame = cap.read()
     
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if tracking_face and i < refresh_size:
                tracking_quality = tracker.update(gray)
                if tracking_quality >= 8.75:
                    t_pos = tracker.get_position()
                    x = int(t_pos.left())
                    y = int(t_pos.top())
                    w = int(t_pos.width())
                    h = int(t_pos.height())
                    i += 1
                else:
                    tracking_face = False

            if not (tracking_face and i < refresh_size):
                i = 0
                rects = detector(gray, 1)
                if rects:
                    rect = rects[0] # We suppose to have a single face detection
                    tracker.start_track(frame, rect)
                    tracking_face = True
        
            if rect:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x,y)-coordinates to a numpy
                # array
                shape = predictor(gray, rect)
                landmarks.append(face_utils.shape_to_np(shape))
                face_rects.append(face_utils.rect_to_bb(rect))

        else:
            break

    cap.release()

    return np.array(landmarks), np.array(face_rects)


def show_face_landmarks(video_filename, predictor_params, full_draw=False, bb_draw=False, frame_draw=True, fps=25.0, out_video_filename="", refresh_size=8):
    """
    Draws facial landmarks over original video frames. If full_draw is True connected lines
    of face landmark points are showed.
    """
    # Convert fps in frame len in milliseconds
    frame_len = int(1000 / fps) if fps > 0 else 0

    # Extract face landmarks and face bounding boxes
    landmarks, face_rects = extract_face_landmarks(video_filename, predictor_params, refresh_size)
    
    cap = cv2.VideoCapture(video_filename)
    if out_video_filename:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(out_video_filename, fourcc, vfps, (width, height))

    for shape, rect in zip(landmarks, face_rects):
        ret, frame = cap.read()
        if not frame_draw:
            frame = np.ones_like(frame) * 255
    
        if ret == True:
            # draw face bounding box
            if bb_draw:
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if full_draw:
                for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
                    # grab the (x, y)-coordinates associated with the
                    # face landmark
                    (j, k) = FACIAL_LANDMARKS_IDXS[name]
                    pts = shape[j:k]
 
                    if name in ('jaw', 'right_eyebrow', 'left_eyebrow'):
                        # since the jawline is a non-enclosed facial region,
                        # just draw lines between the (x, y)-coordinates
                        for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, (100, 200, 0), 1)
                    if name in ('right_eye', 'left_eye'):
                        for l in range(len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, (100, 200, 0), 1)
                    if name == 'nose':
                        for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, (100, 200, 0), 1)
                        cv2.line(frame, tuple(pts[-1]), tuple(pts[3]), (100, 200, 0), 1)
                    if name == 'mouth':
                        for l in range(0, 11):
                            ptA = tuple(pts[l])
                            ptB = tuple(pts[l + 1])
                            cv2.line(frame, ptA, ptB, (100, 200, 0), 1)
                        cv2.line(frame, tuple(pts[0]), tuple(pts[11]), (100, 200, 0), 1)
                        for l in range(12, len(pts) - 1):
                            ptA = tuple(pts[l])
                            ptB = tuple(pts[l + 1])
                            cv2.line(frame, ptA, ptB, (100, 200, 0), 1)
                        cv2.line(frame, tuple(pts[12]), tuple(pts[-1]), (100, 200, 0), 1)
         
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (100, 100, 255), -1)

            # show the output image with face landmarks
            cv2.imshow("Output", frame)
            if out_video_filename:
                out.write(frame)
            cv2.waitKey(frame_len)
        else:
            break

    cap.release()
    if out_video_filename:
        out.release()
    cv2.destroyAllWindows()


def save_face_landmarks_speaker(video_path, dest_path, predictor_params, file_ext='mpg', refresh_size=8):
    # Create destination directory if not exists
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    video_filenames = glob(os.path.join(video_path, '*.' + file_ext))
    
    count = 0
    frame_count = 0
    tot_frame_sum = np.zeros((68, 2))
    tot_frame_square_sum = np.zeros((68,2))
    for v_file in video_filenames:
        print('{} - Video file: {}'.format(count, v_file))
        if file_ext != 'txt':
            landmarks, _ = extract_face_landmarks(v_file, predictor_params, refresh_size)
            l_file = os.path.join(dest_path, os.path.basename(v_file).replace('.' + file_ext, '.npy'))
            #np.savetxt(l_file, landmarks.reshape([-1, 136]), fmt='%d')
            np.save(l_file, landmarks)
        else:
            landmarks = np.load(v_file)
        
        # Compute and save motion vector
        feat = get_motion_vector(landmarks, delta=1)
        #ml_file = os.path.join(dest_path, os.path.basename(v_file).replace('.' + file_ext, '.npy'))
        
        # Update sums
        count += 1
        tot_frame_sum += feat.sum(axis=0)
        tot_frame_square_sum += (feat ** 2).sum(axis=0)
        frame_count += len(feat)

    print('Done. Face landmark files created:', count)
    print('Computing mean and standard deviation of features...')
    # Compute mean and standard deviation of features
    print('Total number of frames:', frame_count)
    feat_mean = tot_frame_sum / frame_count
    feat_std = np.sqrt(tot_frame_square_sum / frame_count - feat_mean ** 2)
    print('done.')

    # Save mean and standard deviation
    np.save(os.path.join(dest_path, 'video_feat_mean.npy'), feat_mean)
    np.save(os.path.join(dest_path, 'video_feat_std.npy'), feat_std)
    print('Normalization data files saved.')


def save_face_landmarks(dataset_path, speakers_list, video_dir, dest_dir, predictor_params, file_ext='mpg', refresh_size=8):
    # Every refresh_size frames a new face detection is forced (the face correlation tracker is ignored).

    for s in speakers_list:
        print('Computing face landmarks of speaker {:d}...'.format(s))
        video_path = os.path.join(dataset_path, 's' + str(s), 's' + str(s) + '.' + video_dir)
        dest_path = os.path.join(dataset_path, 's' + str(s), 's' + str(s) + '.' + dest_dir)
        
        save_face_landmarks_speaker(video_path, dest_path, predictor_params, file_ext, refresh_size)

        print('Speaker {:d} completed.'.format(s))


if __name__ == '__main__':
    dataset_path = 'C:\\Users\\Public\\aau_data\\GRID'
    speakers_list = [11]
    video_dir = 'mpg_vcd'
    dest_dir = 'landmarks'
    predictor_params = 'C:\\Users\\Giovanni Morrone\\Documents\\Dottorato di Ricerca\\Speech Processing\\datasets\\GRID\\shape_predictor_68_face_landmarks.dat'
    file_ext = 'mpg'

    save_face_landmarks(dataset_path, speakers_list, video_dir, dest_dir, predictor_params, file_ext)