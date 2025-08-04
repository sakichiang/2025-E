# calibration.py
# 相机标定工具 - 用来校正镜头畸变，提高图像处理精度
#
# 使用方法：
# 1. 准备一个棋盘格标定板（建议打印在硬纸板上）
# 2. 运行这个程序，对着标定板拍照
# 3. 程序会自动计算相机参数并保存到文件
# 4. 其他程序可以加载这些参数来消除镜头畸变

import cv2
import numpy as np
import os

# ==================== 标定参数配置 ====================
# 棋盘格规格 - 注意这里是内角点数量，不是格子数量
# 比如12x9的棋盘格，内角点是11x8
CHESSBOARD_CORNERS_X = 11   # 水平方向内角点数量
CHESSBOARD_CORNERS_Y = 8    # 垂直方向内角点数量
SQUARE_SIZE_MM = 20         # 每个格子的边长（毫米）

# 标定设置
NUM_IMAGES_TO_CAPTURE = 25  # 需要拍摄的标定图片数量，越多越准确
CAMERA_INDEX = 3            # 摄像头编号，根据你的系统调整

# 输出文件
OUTPUT_FILE = 'camera_calibration.npz'  # 标定结果保存文件
# ======================================================

def main():
    """
    相机标定主程序
    会打开摄像头，引导用户拍摄标定图片，然后计算标定参数
    """
    # 角点检测的精度控制参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备标定板的3D坐标
    # 假设标定板放在z=0平面上，使用实际的物理尺寸（毫米）
    objp = np.zeros((CHESSBOARD_CORNERS_Y * CHESSBOARD_CORNERS_X, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_X, 0:CHESSBOARD_CORNERS_Y].T.reshape(-1, 2) * SQUARE_SIZE_MM

    # 存储所有标定图片的3D点和2D点
    objpoints = []  # 真实世界的3D坐标
    imgpoints = []  # 图像中的2D坐标

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {CAMERA_INDEX}")
        return

    print("\n=== 相机标定程序启动 ===")
    print(f"请准备一个 {CHESSBOARD_CORNERS_X+1}x{CHESSBOARD_CORNERS_Y+1} 的棋盘格")
    print(f"格子大小应该是 {SQUARE_SIZE_MM}毫米")
    print("操作说明：")
    print("  - 按 's' 键保存当前画面用于标定")
    print("  - 按 'q' 键退出程序")
    print("建议从不同角度、不同距离拍摄标定板，这样标定效果更好")

    captured_count = 0
    while captured_count < NUM_IMAGES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头读取图像")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在图像上显示进度和说明
        progress_text = f"已拍摄: {captured_count}/{NUM_IMAGES_TO_CAPTURE}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "按 's' 保存 | 按 'q' 退出", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        # The window title is now in English
        cv2.imshow('Calibration - Press s to save, q to quit', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n用户取消标定")
            break
        elif key == ord('s'):
            # 寻找棋盘格角点
            found, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_X, CHESSBOARD_CORNERS_Y), None)

            if found:
                captured_count += 1
                objpoints.append(objp)

                # 亚像素精度优化角点位置
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # 画出检测到的角点，给用户反馈
                cv2.drawChessboardCorners(frame, (CHESSBOARD_CORNERS_X, CHESSBOARD_CORNERS_Y), corners2, found)
                cv2.imshow('Calibration - Press s to save, q to quit', frame)
                print(f"成功捕获第 {captured_count} 张标定图片")
                print("Could not find chessboard corners in the current frame. Please adjust position and angle.")
            else:
                print("未检测到棋盘格，请调整角度和位置")

    cap.release()
    cv2.destroyAllWindows()

    # 检查是否有足够的标定图片
    if len(imgpoints) < 5:
        print(f"\n标定图片不足（需要至少5张，实际{len(imgpoints)}张），无法进行标定")
        return

    print(f"\n收集了 {len(imgpoints)} 张有效图片，开始计算标定参数...")

    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("标定成功！")
        print("\n相机内参矩阵:")
        print(camera_matrix)
        print("\n畸变系数:")
        print(dist_coeffs)

        # 保存标定结果
        np.savez(OUTPUT_FILE,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                square_size=SQUARE_SIZE_MM)
        print(f"\n标定结果已保存到 '{OUTPUT_FILE}'")

        # 计算重投影误差（标定精度指标）
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        avg_error = mean_error / len(objpoints)
        print(f"平均重投影误差: {avg_error:.3f} 像素")
        if avg_error < 0.5:
            print("标定质量: 优秀")
        elif avg_error < 1.0:
            print("标定质量: 良好")
        else:
            print("标定质量: 一般，建议重新标定")

    else:
        print("标定失败，请检查标定图片质量")

if __name__ == '__main__':
    main()