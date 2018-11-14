#include <iostream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace xfeatures2d;



int ReadPalIm(const String name1, const String name2, Mat& im1, Mat& im2)
{
	im1 = imread(name1);
	if (im1.empty())
	{
		printf("can not load image \n");
		system("pause");
		return -1;
	}
	im2 = imread(name2);
	if (im2.empty())
	{
		printf("can not load image \n");
		system("pause");
		return -1;
	}
	return 0;
}

int HomoSURF(const Mat& _in_1, const Mat& _in_2, Mat& _out_homo)
{
	//-- 初始化
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	int minHessian = 700;                           //定义Hessian矩阵阈值特征点检测算子      
	Ptr<SURF> detector = SURF::create(minHessian);  //定义SURF检测器

	detector->detect(_in_1, keypoints_1);			//调用detect检测特征点
	detector->detect(_in_2, keypoints_2);

	//-- 计算描述子
	detector->compute(_in_1, keypoints_1, descriptors_1);
	detector->compute(_in_2, keypoints_2, descriptors_2);

	//Mat outimg1;
	//drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//-- 描述子匹配
	vector<DMatch>matchePoints;
	FlannBasedMatcher matcher;
	matcher.match(descriptors_1, descriptors_2, matchePoints, Mat());

	//-- 筛选,寻找正确匹配
	sort(matchePoints.begin(), matchePoints.end()); //特征点排序

	vector<DMatch> good_matches;
	vector<Point2f> good_keypoints_1, good_keypoints_2;
	for (int i = 0; i < (int)(matchePoints.size()*0.25); i++)
	{
		good_keypoints_1.push_back(keypoints_1[matchePoints[i].queryIdx].pt);
		good_keypoints_2.push_back(keypoints_2[matchePoints[i].trainIdx].pt);
	}

	_out_homo = findHomography(good_keypoints_1, good_keypoints_2, CV_RANSAC);
	cout << "变换矩阵为：\n" << _out_homo << endl; //输出映射矩阵	
	return 0;
}

int ShowRes(const Mat& _in_1, const Mat& _in_2, const Mat& _homo, bool save = false, const String fname = "_r.jpg")
{
	Mat imageTransform1, imageTransform2; //图像配准
	warpPerspective(_in_1, imageTransform1, _homo, Size(_in_2.cols, _in_2.rows));

	Mat dst1, dst2;
	double alpha = 0.5;
	double beta = 1 - alpha;
	addWeighted(_in_1, alpha, _in_2, beta, 0.0, dst1);
	addWeighted(_in_2, alpha, imageTransform1, beta, 0.0, dst2);

	namedWindow("真实叠加图", 0);
	resizeWindow("真实叠加图", _in_1.cols / 2, _in_1.rows / 2);
	namedWindow("变换叠加图", 0);
	resizeWindow("变换叠加图", _in_1.cols / 2, _in_1.rows / 2);

	imshow("真实叠加图", dst1);
	imshow("变换叠加图", dst2);

	cout << _in_1.rows << ' ' << _in_1.cols;
	waitKey(0);
	destroyWindow("变换叠加图");
	destroyWindow("真实叠加图");
	system("pause");

	if (save == true)
	{
		//imwrite(fname + ".jpg", dst1);
		imwrite(fname , dst2);
	}
	return 0;
}


int main()
{	
	String pic_name1, pic_name2 ,outname;
	Mat pic1, pic2;
	Mat homo;

#if 1
	pic_name1 = "E:\\data\\Result\\15.jpg";
	pic_name2 = "E:\\data\\Result\\16.jpg";
	outname = "E:\\data\\Result\\16_H.jpg";
#else
	cout << "pic1 path and name" << endl;
	cin >> pic_name1;
	cout << "pic2 path and name" << endl;
	cin >> pic_name2;
#endif
	ReadPalIm(pic_name1, pic_name2, pic1, pic2);
	HomoSURF(pic1, pic2, homo);
	ShowRes(pic1, pic2, homo, true, outname);
	return 0;
}