import cv2
import numpy as np
import math
import glob
from geographiclib.geodesic import Geodesic, GeodesicCapability
import webbrowser

# Calculate key points based on features with Lucas-Kanade algorithm
def Calkeypoints(image_ref, image_cur, prev_KP):

    #Only the keypoints are interesting
    kp2, _, _ = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, prev_KP, None, winSize=(15,15), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kp1, _, _ = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, winSize=(15,15), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Verify the difference between key points to identify valid result
    d = abs(prev_KP - kp1).reshape(-1, 2).max(-1)
    valid = d < 1

    # Create new lists with the good features
    n_kp1, n_kp2 = [], []
    for i, j in enumerate(valid):
        if j:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    # Format the features into float32 numpy arrays
    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

    return n_kp1, n_kp2


# Draw optical flow for key points origins
def drawFlow(img, prev_img, curr_img):

    #Draw arrow from the key points showing the optical flow
    for i, (j, k) in enumerate(zip(curr_img, prev_img)):
        a, b = k.ravel()
        v = tuple((j - k) * 2.5 + k)
        v = tuple([int(_) for _ in v])
        d = [j - k][0] * 0.75

        tip1 = tuple(np.float32(np.array([a, b]) + rotate([d], 0.5))[0])
        tip2 = tuple(np.float32(np.array([a, b]) + rotate([d], -0.5))[0])
        tip1 = tuple([int(_) for _ in tip1])
        tip2 = tuple([int(_) for _ in tip2])

        cv2.line(img, v, (int(a), int(b)), (0, 255, 0),2)
        cv2.line(img, (int(a), int(b)), tip1, (0, 0, 255),2)
        cv2.line(img, (int(a), int(b)), tip2, (0, 0, 255),2)

    cv2.imshow('Real time', img)

# Apply Rotation matrix
def rotate(points, theta):

    # Matrice de Rotation
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)] ])
    rot = []
    for v in points:
        v = np.array(v).transpose()
        v = R.dot(v)
        v = v.transpose()
        rot.append(v)

    return rot

# Draw the real time trajectory
def Trajectory(image, x, y):

    #Indicate origin
    cv2.putText(image, 'Origin (0,0)', (260, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # Draw the current location
    cv2.circle(image, (int(x) + 300, int(y) + 50), 1, (255,255,255))

    # Erase Previous Text
    cv2.rectangle(image, (50, 450), (450, 600), (0, 0, 0), -1)

    # Current Location Coordinates
    text = "Location: x=%.2f m y=%.2f m " % (x, y)
    cv2.putText(image, text, (50, 480), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # Y axis
    cv2.arrowedLine(image, (500, 500), (500, 550), (255, 255, 255), 1)
    cv2.putText(image, 'y axis', (440, 510), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # X axis
    cv2.arrowedLine(image, (500, 500), (550, 500), (255, 255, 255), 1)
    cv2.putText(image, 'x axis', (500, 490), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    #cv2.imshow('Trajectoire', image)
    image = drawError(image, x, y)

    return image

def main():

    #Initialisation variables
    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    i = 0
    CameraMatrix = np.float64([[718.856, 0, 607.1929], [0, 718.856, 185.2157], [0, 0, 1]])
    T_vectors = []
    R_matrices = []
    R = None
    T = None
    current_KP = None
    Detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    skip = 0

    # Loop across all images
    for filepath in glob.iglob('images/*.jpg'):

        # Image to show on screen
        originalImage = cv2.imread(filepath)

        #Image for computation in black & white
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

        # Optimization comment or uncomment to see difference
        # Create a CLAHE object (contrast limiting adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0)
        grayImage = clahe.apply(grayImage)

        # Optical Flow Calculation
        if i == 0:
            # Detect new features
            features = Detector.detect(grayImage)
            current_KP = np.array([x.pt for x in features], dtype=np.float32)

            # Zeroing the first T and R matrix
            T_vectors.append(tuple([[0], [0], [0]]))
            R_matrices.append(tuple(np.zeros((3, 3))))

        elif i == 1:

            # Obtain feature correspondence points
            prev_KP, current_KP = Calkeypoints(prev_img, grayImage, prev_KP)

            # Calculate the Essential Matrix - LMEDS vs RANSAC
            E, _ = cv2.findEssentialMat(current_KP, prev_KP, CameraMatrix, method=cv2.LMEDS, prob=0.999, threshold=1.0)

            # Recover the translation and rotation matrix
            _, R,  T, _ = cv2.recoverPose(E, current_KP, prev_KP, CameraMatrix)

            T_vectors.append(tuple(R.dot(T)))
            R_matrices.append(tuple(R))

        else:
            # Obtain Key features points
            prev_KP, current_KP = Calkeypoints(prev_img, grayImage, prev_KP)
            # Verify if the current frame is going to be skipped

            # Calculate the Essential Matrix - LMEDS vs RANSAC
            E, _ = cv2.findEssentialMat(current_KP, prev_KP, CameraMatrix, method=cv2.LMEDS, prob=0.999, threshold=1.0)

            # Recover the translation and rotation matrix
            _, R, t, _ = cv2.recoverPose(E, current_KP, prev_KP, CameraMatrix)

            T = prev_T + prev_R.dot(t)
            R = R.dot(prev_R)
            T_vectors.append(tuple(T))
            R_matrices.append(tuple(R))

            # TODO Check if needed
            if prev_KP.shape[0] < 1000:  # Verify if the amount of feature points
                skip = 1
                features = Detector.detect(grayImage)
                current_KP = np.array([x.pt for x in features], dtype=np.float32)

        if i == 0:
            x = 0
            y = 0
        else:
            x = T[0]
            y = T[1]

            #This logic solve a graphical displeasure not relevant for calculation, only visual
            if not skip:
                drawFlow(originalImage, prev_KP, current_KP)
            else:
                skip = 0
                drawFlow(originalImage, prev_KP, prev_KP)

        # Draw Trajectory and saving the image for next update
        traj = Trajectory(traj, x, y)

        # Keep current frame for next frame calculation
        i += 1
        prev_img = grayImage
        prev_R = R
        prev_T = T
        prev_KP = current_KP

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def drawError(traj, x, y):

    # For google map request
    global web_lat
    global web_lon

    # GPS ground truth coordinates
    groundtruth =[
                  [45.50061,-73.61424],
                  [45.50038,-73.61442],
                  [45.50029,-73.61449],
                  [45.50022,-73.61454],
                  [45.50007,-73.61468],
                  [45.49996,-73.61477],
                  [45.49978,-73.61498],
                  [45.49961,-73.61519]]

    # Distance between GPS coordinates from GMAPS **Estimation** (in m)
    KEYPOINT_DIST = 17

    # Radius of the Earth
    R = 6378.0

    # Convert the y distance in km
    d = math.sqrt(x**2+y**2)*0.001

    # Obtaint groundtruth value
    g_v = np.rint(math.sqrt(x**2+y**2)/KEYPOINT_DIST)
    if g_v > 7:
        g_v = 7
    # Initial GPS Coordinates
    lat1_d = (groundtruth[0][0])
    lon1_d = (groundtruth[0][1])

    # Estimate of the current GPS Coordinates from groundtruth
    g_lat = (groundtruth[int(g_v)][0])
    g_lon =(groundtruth[int(g_v)][1])

    #  Calculate the groundtruth bearing from GPS Coordinates, in degree
    g_brng_d = Geodesic.WGS84.Inverse(lat1_d, lon1_d, g_lat, g_lon)['azi1']

    # Deg to Rad for calculation
    g_brng_r = math.radians(g_brng_d)
    lat1 = math.radians(lat1_d)
    lon1 = math.radians(lon1_d)

    # Calculate next GPS coordinates from displacement
    lat2 = math.asin(math.sin(lat1)*math.cos(d/R) + math.cos(lat1)*math.sin(d/R)*math.cos(g_brng_r))
    lon2 = lon1 + math.atan2(math.sin(g_brng_r)*math.sin(d/R)*math.cos(lat1), math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

    # Rad to Deg for interpretation on gmap
    lat2 = round(math.degrees(lat2),6)
    lon2 = round(math.degrees(lon2),6)

    # Calculate the error between the groundtruth and estimate GPS Coordinates
    err_lat = abs((lat2-g_lat)/lat2)*100
    err_lon = abs((lon2-g_lon)/lon2)*100

    # Estimate distance from groundtruth and estimated GPS coordinates
    gt_g = (Geodesic.WGS84.Inverse(lat1_d, lon1_d, g_lat, g_lon, outmask=GeodesicCapability.DISTANCE))
    g = (Geodesic.WGS84.Inverse(lat1_d, lon1_d, lat2, lon2, outmask = GeodesicCapability.DISTANCE))

    web_lat = lat2
    web_lon = lon2

    #  Calculate the estimate bearing from GPS Coordinates, in degree
    e_brng_d = Geodesic.WGS84.Inverse(lat1_d, lon1_d, lat2, lon2)['azi1']
    err_brng = abs((e_brng_d-g_brng_d)/e_brng_d)*100
    err_t = abs((e_brng_d-g_brng_d))

    # calculate error on distance
    # Absolute error
    err_abs = abs(math.sqrt(x**2+y**2) - g['s12'])

    #Division by 0 protection
    if g['s12'] == 0:
        err_pourc = float('nan')
    else:
        err_pourc = abs((math.sqrt(x**2+y**2) - g['s12'])/g['s12'])*100

    #Ground truth GPS error
    g_err_abs = abs(math.sqrt(x**2+y**2) - gt_g['s12'])
    #Division by 0 protection
    if gt_g['s12'] == 0:
        g_err_pourc = float('nan')
    else:
        g_err_pourc = abs((math.sqrt(x**2+y**2) - gt_g['s12'])/gt_g['s12'])*100

    text = "Erreur GPS_distance: %.4f m (%.4f %%)" % (err_abs, err_pourc)
    cv2.putText(traj, text, (50, 500), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    text = "Erreur latitude: %.10f  %%" % float(err_lat)
    cv2.putText(traj, text, (50, 520), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    text = "Erreur longitude: %.10f %%" % float(err_lon)
    cv2.putText(traj, text, (50, 540), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    text = "Erreur bearing: %.2f %%" % err_brng
    cv2.putText(traj, text, (50, 560), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    text = "Erreur GT_distance: %.4f m (%.4f %%)" % (g_err_abs, g_err_pourc)
    cv2.putText(traj, text, (50, 580), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.imshow('Trajectoire', traj)

    return traj

if __name__ == '__main__':
    main()
    #Pin point the final GPS Coordinate
    #webbrowser.open("https://www.google.com/maps/place/" + str(web_lat) + "," + str(web_lon))
    webbrowser.open("http://maps.google.com/maps/place/"+str(web_lat) +","+ str(web_lon))
    cv2.waitKey()