import unittest

from pkg.pose.mediapipe_pose import MediaPipePose
from pkg.video_reader.video_reader import VideoReader

video_path = '../../dataset/dataset_1/archive/barbell biceps curl/barbell biceps curl_44.mp4'


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.sample_video_reader = VideoReader(video_path)
        self.pose = MediaPipePose(self.sample_video_reader)

    def test_calculateAngle(self):
        angle = self.pose.calculateAngle(
            (558, 326, 0),
            (642, 333, 0),
            (718, 321, 0)
        )
        expected_angle = 166
        self.assertEqual(expected_angle, int(angle))  # add assertion here

    def test_extract_keypoints(self):
        keypoints = self.pose.extract_keypoints()
        print(keypoints)
        self.assertTrue(len(keypoints) > 0)
        self.assertTrue(len(keypoints[0]) > 0)
        self.assertTrue(len(keypoints[0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0][0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0][0][0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0][0][0][0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0][0][0][0][0][0]) > 0)
        self.assertTrue(len(keypoints[0][0][0][0][0][0][0][0][0]) > 0)


if __name__ == '__main__':
    unittest.main()
