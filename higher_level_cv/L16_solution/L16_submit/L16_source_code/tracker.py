import numpy as np
from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from PIL import ImageDraw,Image,ImageFont


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = KalmanFilter(prediction)  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.counted = False
        self.trace = []  # trace path
        self.ground_truth_box = self.prediction


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, iou_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            iou_thresh: iou threshold. When smaller than the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.iou_thresh = iou_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.TOP=350
        self.BOTTOM=900


    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # A is the top-lelft point and B is the bottom-right
        boxA=boxA.reshape((4,))
        boxB=boxB.reshape((4,))
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def Update(self,r_image, detections):
        draw=ImageDraw.Draw(r_image)
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=40)
        draw.line([(0,self.TOP),(720,self.TOP)],fill='rgb(0,0,255)',width=3)
        draw.line([(0,self.BOTTOM),(720,self.BOTTOM)],fill='rgb(0,0,255)',width=3)
        # Create tracks if no tracks vector found
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                draw.text(detections[i][:2],str(track.track_id),fill='rgb(0,255,0)',font=font)
                print('- init track:',track.track_id)
                self.trackIdCount += 1
                self.tracks.append(track)
            # return r_image

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                # try:
                iou = self.bb_intersection_over_union(self.tracks[i].prediction, detections[j])
                cost[i][j] = iou
                # except:
                #     pass
        # print('Cost: ', cost)

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(-cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] < self.iou_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    print('Delete track over max_frame_to_skip',self.tracks[id].track_id)
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment: 
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in un_assigned_detects:
                # if the object lie on too high along y axis, it seem to be not the first time it appear
                if (detections[i][1]+(detections[i][3]-detections[i][1])/2) > self.TOP+100:
                    continue
                track = Track(detections[i], self.trackIdCount)
                draw.text(detections[i][:2],str(track.track_id),fill='rgb(0,255,0)',font=font)
                print('- add new track:',track.track_id)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                draw.text(detections[assignment[i]][:2],str(self.tracks[i].track_id),fill='rgb(0,255,0)',font=font)
                print('- assigned track:',self.tracks[i].track_id)
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                    detections[assignment[i]], 1)
                self.tracks[i].ground_truth_box = detections[assignment[i]]
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                    [[0.], [0.], [0.], [0.]], 0)
                self.tracks[i].ground_truth_box = self.tracks[i].prediction

            if len(self.tracks[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            cx, cy, ux, uy = self.tracks[i].prediction[0][0], self.tracks[i].prediction[0][1], \
                             self.tracks[i].prediction[0][2], self.tracks[i].prediction[0][3]
            self.tracks[i].trace.append([[(cx + ux)/2], [(cy + uy)/2]])
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
        font2 = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=45)
        draw.text((0,0),'COUNT: '+str(self.trackIdCount-1),fill='rgb(0,255,0)',font=font2) 
        return r_image
