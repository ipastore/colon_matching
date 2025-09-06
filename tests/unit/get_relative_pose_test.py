import numpy as np
import unittest

def get_relative_pose(R1, t1, R2, t2):
    """ Compute the relative pose (R, t) from COLMAP. New reference frame is image1 """
    R_rel = R1.T @ R2  
    t_rel = R1.T @ (t2 - t1)  
    return R_rel, t_rel

class TestRelativePose(unittest.TestCase):
    
    def test_only_translation(self):
        """ Test case where only translation exists (no rotation). """
        R1 = np.eye(3)  # Identity matrix (no rotation)
        t1 = np.array([1, 2, 3])  # Some arbitrary translation

        R2 = np.eye(3)  # Identity matrix (no rotation)
        t2 = np.array([4, 5, 6])  # Different translation

        expected_R_rel = np.eye(3)  # No rotation change
        expected_t_rel = np.array([3, 3, 3])  # t2 - t1 in world frame

        R_rel, t_rel = get_relative_pose(R1, t1, R2, t2)
        
        np.testing.assert_array_almost_equal(R_rel, expected_R_rel)
        np.testing.assert_array_almost_equal(t_rel, expected_t_rel)

    def test_only_rotation(self):
        """ Test case where only rotation exists (no translation). """
        R1 = np.eye(3)  # Identity matrix (no rotation)
        t1 = np.zeros(3)  # No translation

        R2 = np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 1]])  # Rotation matrix for 90 degrees about Z-axis
        t2 = np.zeros(3)  # No translation

        expected_R_rel = R2  # Since R1 is identity, R_rel should be R2
        expected_t_rel = np.zeros(3)  # No translation difference

        R_rel, t_rel = get_relative_pose(R1, t1, R2, t2)
        
        np.testing.assert_array_almost_equal(R_rel, expected_R_rel)
        np.testing.assert_array_almost_equal(t_rel, expected_t_rel)

    def test_rotation_and_translation(self):
        """ Test case where both rotation and translation exist. """
        R1 = np.eye(3)  # No rotation for first pose
        t1 = np.array([1, 0, 0])  # Positioned at x=1
        
        R2 = np.array([[0, -1, 0],  # 90-degree rotation about Z-axis
                       [1,  0, 0],
                       [0,  0, 1]])
        t2 = np.array([1, 1, 0])  # Positioned at x=1, y=1

        expected_R_rel = R2  # Should be 90-degree rotation matrix
        expected_t_rel = np.array([0, 1, 0])  # In the rotated frame, translation should be along Y

        R_rel, t_rel = get_relative_pose(R1, t1, R2, t2)

        np.testing.assert_array_almost_equal(R_rel, expected_R_rel)
        np.testing.assert_array_almost_equal(t_rel, expected_t_rel)
    
        
if __name__ == '__main__':
    unittest.main()