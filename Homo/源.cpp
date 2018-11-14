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
	//-- ��ʼ��
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	int minHessian = 700;                           //����Hessian������ֵ������������      
	Ptr<SURF> detector = SURF::create(minHessian);  //����SURF�����

	detector->detect(_in_1, keypoints_1);			//����detect���������
	detector->detect(_in_2, keypoints_2);

	//-- ����������
	detector->compute(_in_1, keypoints_1, descriptors_1);
	detector->compute(_in_2, keypoints_2, descriptors_2);

	//Mat outimg1;
	//drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//-- ������ƥ��
	vector<DMatch>matchePoints;
	FlannBasedMatcher matcher;
	matcher.match(descriptors_1, descriptors_2, matchePoints, Mat());

	//-- ɸѡ,Ѱ����ȷƥ��
	sort(matchePoints.begin(), matchePoints.end()); //����������

	vector<DMatch> good_matches;
	vector<Point2f> good_keypoints_1, good_keypoints_2;
	for (int i = 0; i < (int)(matchePoints.size()*0.25); i++)
	{
		good_keypoints_1.push_back(keypoints_1[matchePoints[i].queryIdx].pt);
		good_keypoints_2.push_back(keypoints_2[matchePoints[i].trainIdx].pt);
	}

	_out_homo = findHomography(good_keypoints_1, good_keypoints_2, CV_RANSAC);
	cout << "�任����Ϊ��\n" << _out_homo << endl; //���ӳ�����	
	return 0;
}

int ShowRes(const Mat& _in_1, const Mat& _in_2, const Mat& _homo, bool save = false, const String fname = "_r.jpg")
{
	Mat imageTransform1, imageTransform2; //ͼ����׼
	warpPerspective(_in_1, imageTransform1, _homo, Size(_in_2.cols, _in_2.rows));

	Mat dst1, dst2;
	double alpha = 0.5;
	double beta = 1 - alpha;
	addWeighted(_in_1, alpha, _in_2, beta, 0.0, dst1);
	addWeighted(_in_2, alpha, imageTransform1, beta, 0.0, dst2);

	namedWindow("��ʵ����ͼ", 0);
	resizeWindow("��ʵ����ͼ", _in_1.cols / 2, _in_1.rows / 2);
	namedWindow("�任����ͼ", 0);
	resizeWindow("�任����ͼ", _in_1.cols / 2, _in_1.rows / 2);

	imshow("��ʵ����ͼ", dst1);
	imshow("�任����ͼ", dst2);

	cout << _in_1.rows << ' ' << _in_1.cols;
	waitKey(0);
	destroyWindow("�任����ͼ");
	destroyWindow("��ʵ����ͼ");
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