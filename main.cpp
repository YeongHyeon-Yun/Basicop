#if 0

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();


int main()
{
	PointOp();
}

void PointOp()
{
	Point pt1;
	pt1.x = 5; pt1.y = 10;
	Point pt2(10, 30);

	Point pt3 = pt1 + pt2; // pt1.operator + (pt2) or operator + (pt1 ,pt2)
	Point pt4 = pt1 * 2; // pt1.operator * (2) or operator * (pt1, 2)
	int d1 = pt1.dot(pt2);
	bool b1 = (pt1 == pt2);

	cout << "pt1: " << pt1 << endl;
	cout << "pt2: " << pt1 << endl;
}

#endif

#if 0

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();
void SizeOp();

int main()
{
	SizeOp();
}

void SizeOp()
{
	Size sz1, sz2(10, 20);
	sz1.width = 5; sz1.height = 10;

	Size sz3 = sz1 + sz2; // operator + (sz1, sz2)
	Size sz4 = sz1 * 2;

	int area1 = sz4.area();

	cout << "sz3" << sz3 << endl;
	cout << "sz4" << sz4 << endl;
}

#endif

#if 0

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();
void SizeOp();
void RectOp();

int main()
{
	RectOp();
}

void RectOp()
{
	Rect rc1;
	Rect rc2(10, 10, 60, 40);
	Rect rc3 = rc1 + Size(50, 40);
	Rect rc4 = rc2 + Point(10, 10);

	Rect rc5 = rc3 & rc4;
	Rect rc6 = rc3 | rc4;

	cout << "rc5" << rc5 << endl;
	cout << "rc6" << rc6 << endl;
}

#endif

#if 0

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();
void SizeOp();
void RectOp();
void RotateRectOp();

int main()
{
	RectOp();
	RotateRectOp();
}

void RotateRectOp()
{
	/*RotatedRect rr1(Point2f(40, 30), Size2f(40, 20), 30.f);
	Point2f pts[4];
	rr1.points(pts);

	cout << "pts[0]: " << pts[0] << ", pts[1]: " << pts[1] << ", pts[2]: " << pts[2] << ", pts[3]: " << pts[3] << ", pts[4]: " << pts[4] << endl;*/
	Mat test_image(200, 200, CV_8UC3, Scalar(0));
	RotatedRect rRect = RotatedRect(Point2f(100, 100), Size2f(100, 50), 30.f);

	Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(test_image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

	Rect brect = rRect.boundingRect();
	rectangle(test_image, brect, Scalar(255, 0, 0), 2);
	imshow("rectangle", test_image);
	waitKey(0);

}

void RectOp()
{
	Rect rc1;
	Rect rc2(10, 10, 60, 40);
	Rect rc3 = rc1 + Size(50, 40);
	Rect rc4 = rc2 + Point(10, 10);

	Rect rc5 = rc3 & rc4;
	Rect rc6 = rc3 | rc4;

	cout << "rc5" << rc5 << endl;
	cout << "rc6" << rc6 << endl;
}
#endif

#if 0

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();
void SizeOp();
void RectOp();
void RotateRectOp();
void RangeOp();

int main()
{
	RangeOp();
}

void RangeOp()
{
	Mat img;
	Mat rect_img;
	img = imread("lenna.bmp");

	int y1 = 220;
	int y2 = 320;
	int x1 = 200;
	int x2 = 370;

	if (img.empty()) {
		cerr << "Image load failed! " << endl;
		return;
	}
	rect_img = img(Range(y1, y2), Range(x1, x2));
	namedWindow("원본이미지", WINDOW_NORMAL);
	namedWindow("결과이미지", WINDOW_AUTOSIZE);
	imshow("원본이미지", img);
	imshow("결과이미지", rect_img);
	waitKey();

	destroyAllWindows();

}

void RotateRectOp()
{
	/*RotatedRect rr1(Point2f(40, 30), Size2f(40, 20), 30.f);
	Point2f pts[4];
	rr1.points(pts);

	cout << "pts[0]: " << pts[0] << ", pts[1]: " << pts[1] << ", pts[2]: " << pts[2] << ", pts[3]: " << pts[3] << ", pts[4]: " << pts[4] << endl;*/
	Mat test_image(200, 200, CV_8UC3, Scalar(0));
	RotatedRect rRect = RotatedRect(Point2f(100, 100), Size2f(100, 50), 30.f);

	Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(test_image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

	Rect brect = rRect.boundingRect();
	rectangle(test_image, brect, Scalar(255, 0, 0), 2);
	imshow("rectangle", test_image);
	waitKey(0);

}

void RectOp()
{
	Rect rc1;
	Rect rc2(10, 10, 60, 40);
	Rect rc3 = rc1 + Size(50, 40);
	Rect rc4 = rc2 + Point(10, 10);

	Rect rc5 = rc3 & rc4;
	Rect rc6 = rc3 | rc4;

	cout << "rc5" << rc5 << endl;
	cout << "rc6" << rc6 << endl;
}
#endif

#if 0

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();
void SizeOp();
void RectOp();
void RotateRectOp();
void RangeOp();
void StringOp();

int main()
{
	StringOp();
}

void StringOp()
{
	String str1 = "Hello";
	String str2 = "world";
	String str3 = str1 + " " + str2;

	cout << str3 << endl;

	Mat imgs[3];
	for (int i = 0; i < 3; i++) {
		String filename = format("data%02d.bmp", i + 1);
		cout << filename << endl;
	}
}

void RangeOp()
{
	Mat img;
	Mat rect_img;
	img = imread("lenna.bmp");

	int y1 = 220;
	int y2 = 320;
	int x1 = 200;
	int x2 = 370;

	if (img.empty()) {
		cerr << "Image load failed! " << endl;
		return;
	}
	rect_img = img(Range(y1, y2), Range(x1, x2));
	namedWindow("원본이미지", WINDOW_NORMAL);
	namedWindow("결과이미지", WINDOW_AUTOSIZE);
	imshow("원본이미지", img);
	imshow("결과이미지", rect_img);
	waitKey();

	destroyAllWindows();

}

void RotateRectOp()
{
	/*RotatedRect rr1(Point2f(40, 30), Size2f(40, 20), 30.f);
	Point2f pts[4];
	rr1.points(pts);

	cout << "pts[0]: " << pts[0] << ", pts[1]: " << pts[1] << ", pts[2]: " << pts[2] << ", pts[3]: " << pts[3] << ", pts[4]: " << pts[4] << endl;*/
	Mat test_image(200, 200, CV_8UC3, Scalar(0));
	RotatedRect rRect = RotatedRect(Point2f(100, 100), Size2f(100, 50), 30.f);

	Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(test_image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);

	Rect brect = rRect.boundingRect();
	rectangle(test_image, brect, Scalar(255, 0, 0), 2);
	imshow("rectangle", test_image);
	waitKey(0);

}

void RectOp()
{
	Rect rc1;
	Rect rc2(10, 10, 60, 40);
	Rect rc3 = rc1 + Size(50, 40);
	Rect rc4 = rc2 + Point(10, 10);

	Rect rc5 = rc3 & rc4;
	Rect rc6 = rc3 | rc4;

	cout << "rc5" << rc5 << endl;
	cout << "rc6" << rc6 << endl;
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp1();

int main() {

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	MatOp1();
}

void MatOp1()
{
	Mat img1; // empty matrix
	// 640 X 480
	Mat img2(480, 640, CV_8UC1); //8비트 1채널
	Mat img3(480, 640, CV_8UC3); //8비트 3채널
	Mat img4(Size(640, 480), CV_8UC3);
	Mat img5(480, 640, CV_8UC1, Scalar(200));
	/*
	namedWindow("image1");
	imshow("image1", img5);
	waitKey();
	*/

	Mat img6(480, 640, CV_8UC3, Scalar(150, 150, 0));
	/*

	namedWindow("image2");
	imshow("image2", img6);
	waitKey();
	*/

	Mat mat1 = Mat::zeros(3, 3, CV_8SC1);
	Mat mat2 = Mat::ones(3, 3, CV_32FC1);
	Mat mat3(3, 3, CV_32FC1, Scalar(6.25));
	Mat mat4 = Mat::eye(3, 3, CV_32SC1);

	char data1[] = { 1, 2, 3, 4, 5, 6 };
	Mat mat5(3, 2, CV_8UC1, data1);

	data1[0] = 100;
	data1[3] = 300;

	Mat_<float> mat5_(2, 3);
	mat5_ << 1, 2, 3, 4, 5, 6;
	Mat mat6 = mat5_;

	Mat mat7 = (Mat_<float>(2, 3) << 1, 2, 3, 4, 5, 6);

	mat4.create(5, 5, CV_8UC3);
	mat5.create(4, 4, CV_32FC1);

	mat4 = Scalar(0, 0, 255);
	mat5.setTo(1.f);

}


#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp1();
void MatOp2();

int main() {

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	MatOp2();
}

void MatOp2()
{
	Mat img1 = imread("dog.bmp");

	if (img1.empty()) {
		cerr << "Image load failed! " << endl;
		return;
	}

	Mat img2 = img1; Mat img3;
	img3 = img1;
	namedWindow("img1");
	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	waitKey();
	destroyAllWindows();

	Mat img4 = img1.clone();
	Mat img5;
	img1.copyTo(img5);
	img1.setTo(Scalar(0, 0, 255));

	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	imshow("img4", img4);
	imshow("img5", img5);

	waitKey();
	destroyAllWindows();

}


#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp1();
void MatOp2();
void MatOp3();

int main() {

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	MatOp3();
}

void MatOp3()
{
	Mat img1 = imread("cat.bmp");

	if (img1.empty()) {
		cerr << "Image load failed! " << endl;
		return;
	}

	Mat img2 = img1(Rect(220, 120, 340, 240));
	Mat img3 = img1(Rect(220, 120, 340, 240)).clone();
	/*
		imshow("img1", img1);
		imshow("img2", img2);
		imshow("img3", img3);
		waitKey();
		destroyAllWindows();

		img2 = ~img2;
		imshow("img1", img1);
		imshow("img2", img2);
		imshow("img3", img3);
		waitKey();
		destroyAllWindows();

		Mat mat7(5, 5, CV_8UC1);
		mat7 = Scalar(10);
	*/
	img2 = ~img2;
	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	waitKey();
	destroyAllWindows();


	Mat img4 = img1.rowRange(120, 360);
	Mat img5 = img1.colRange(220, 560);
	Mat img6 = img1.colRange(220, 560);
	imshow("img4", img4);
	imshow("img5", img5);
	imshow("img2", img2);
	imshow("img6", img6);
	waitKey();
	destroyAllWindows();

	Mat mat7 = Mat_<uchar>({ 4, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
	Mat mat8 = mat7.row(2);
	Mat mat9 = mat7.row(2).clone();
	Mat mat10 = mat7.col(3).clone();

}

void MatOp2()
{
	Mat img1 = imread("dog.bmp");

	if (img1.empty()) {
		cerr << "Image load failed! " << endl;
		return;
	}

	Mat img2 = img1; Mat img3;
	img3 = img1;
	namedWindow("img1");
	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	waitKey();
	destroyAllWindows();

	Mat img4 = img1.clone();
	Mat img5;
	img1.copyTo(img5);
	img1.setTo(Scalar(0, 0, 255));

	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	imshow("img4", img4);
	imshow("img5", img5);

	waitKey();
	destroyAllWindows();

}

#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp1();
void MatOp2();
void MatOp3();
void MatOp4();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//MatOp1();
	//MatOp2();
	//MatOp3();
	MatOp4();
}

void MatOp4() {
	Mat mat1 = Mat::zeros(3, 4, CV_8UC1);

	for (int j = 0; j < mat1.rows; j++) {
		for (int i = 0; i < mat1.cols; i++) {
			mat1.at<uchar>(j, i)++;
		}
	}
	for (int j = 0; j < mat1.rows; j++) {
		uchar* p = mat1.ptr<uchar>(j);
		for (int i = 0; i < mat1.cols; i++) {
			p[i]++;
		}
	}
	for (MatIterator_<uchar> it = mat1.begin<uchar>(); it != mat1.end<uchar>(); ++it) {
		(*it)++;
	}
	//위 3개의 for문이 기능이 똑같음

	cout << "mat1:\n" << mat1 << endl;


	Mat img1 = imread("cat.bmp");
	if (img1.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Mat img2 = img1(Rect(220, 120, 340, 240));

	imshow("img1", img1);
	imshow("img2", img2);
	waitKey();
	destroyAllWindows();

	/*
	uchar* p;
	p = img2.data;
	for (; p < img2.dataend; p++)
		*p = ~(*p);
	imshow("img1", img1);
	imshow("img2", img2);
	waitKey();
	destroyAllWindows();
	*/
	uchar* p;
	for (int i = 120; i < 360; i++) {
		p = img1.ptr<uchar>(i);
		p = p + 660;
		for (int j = 0; j < 1020; j++) {
			*(p + j) = ~(*(p + j));
		}
	}
	imshow("img1", img1);
	imshow("img2", img2);
	waitKey();
	destroyAllWindows();

	for (MatIterator_<Vec3b> it = img2.begin<Vec3b>(); it != img2.end<Vec3b>(); it++) {
		//(*it) = ~(*it); ~연산자가 정의되어 있지 않아 사용 불가능
		Vec3b& p1 = (*it);
		p1[0] = ~p1[0];
		p1[1] = ~p1[1];
		p1[2] = ~p1[2];
	}
	imshow("img1", img1);
	imshow("img2", img2);
	waitKey();
	destroyAllWindows();
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp1();
void MatOp2();
void MatOp3();
void MatOp4();
void MatOp5();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//MatOp1();
	//MatOp2();
	//MatOp3();
	//MatOp4();
	MatOp5();
}
void MatOp5() {
	Mat img1 = imread("lenna.bmp");
	cout << "width: " << img1.cols << endl;
	cout << "Height: " << img1.rows << endl;
	cout << "Channels: " << img1.channels() << endl;

	if (img1.type() == CV_8UC1)
		cout << "grayscale" << endl;
	else if (img1.type() == CV_8UC3)
		cout << "color" << endl;
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	MatExpr operator + (const Mat& a, const Mat& b);

	uchar data1[] = { 1, 2, 3, 4 };
	uchar data2[] = { 10, 20, 30, 40 };
	Mat mat1(2, 2, CV_8UC1, data1);
	Mat mat2(2, 2, CV_8UC1, data2);

	Mat mat3;
	mat3 = mat1 + mat2; // operator + (ma1, mat2)

	// Mat Expr operator + (const Mat& a, const Scalar& s):
	mat3 = mat1 + Scalar(10);

	// Mat Expr operator + (const Scalar& s, const Mat& a):
	mat3 = Scalar(10) + mat1;

	mat3 = mat1 - mat2;
	mat3 = mat1 - Scalar(10);
	mat3 = Scalar(10) - mat1;

	mat3 = -mat1;

	//MatExpr operator *(const Mat& a, const Mat& b)
	mat3 = mat1 * mat2;
	mat3 = mat1 * 5;
	mat3 = 5 * mat1;

	mat3 = mat1 / mat2;
	mat3 = mat1 / 5;
	mat3 = 5 / mat1;
}

#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp6();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	MatOp6();
}

void MatOp6()
{
	float data[] = { 1, 1, 2, 3 };
	Mat mat4(2, 2, CV_32FC1, data);
	cout << "mat:\n" << mat4 << endl;

	Mat mat5 = mat4.inv();
	cout << "mat5:\n" << mat5 << endl;

	cout << "mat4.t():\n" << mat4.t() << endl;
	cout << "mat4 * mat5:\n" << mat4 * mat5 << endl;
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void MatOp7();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	MatOp7();
}

void MatOp7() {
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

	Mat img1f;
	img.convertTo(img1f, CV_32FC1);

	uchar data1[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	Mat mat1(3, 4, CV_8UC1, data1);
	Mat mat2 = mat1.reshape(0, 2);

	cout << "mat1:\n" << mat1 << endl;
	cout << "mat2:\n" << mat2 << endl;


	Mat mat3 = Mat::ones(2, 4, CV_8UC1) * 255;
	mat1.push_back(mat3);
	cout << "mat1:\n" << mat1 << endl;
	mat1.pop_back(2);
	cout << "mat1:\n" << mat1 << endl;

	mat1.resize(6, 100);
	cout << "mat1:\n" << mat1 << endl;
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace std;

void VecOp();

int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

	VecOp();
}

void VecOp() {

	Vec3b p1, p2(0, 0, 255);
	p1[0] = 100;
	p1.val[1] = 10;
	cout << "p1: " << p1 << endl;
	cout << "p2: " << p2 << endl;

	Scalar gray = 128;
	gray.val[0] = 100;
	cout << "gray: " << gray << endl;

	Scalar yellow(0, 255, 255);
	cout << "yellow: " << yellow << endl;

}
#endif

#if 0
#include "opencv2//opencv.hpp"
#include <iostream>
#include "opencv2/core/utils/logger.hpp"
using namespace cv;
using namespace std;
void camera_in();
void video_in();
void camera_in_video_out();
int main() {
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	//camera_in();
	//video_in();
	camera_in_video_out();
}
void camera_in_video_out()
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return;
	}
	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(CAP_PROP_FPS);
	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
	int delay = cvRound(1000 / fps);
	VideoWriter outputVideo("output.mp4", fourcc, fps, Size(w, h));
	if (!outputVideo.isOpened()) {
		cout << "File open failed!" << endl;
		return;
	}
	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		inversed = ~frame;
		outputVideo << inversed;
		imshow("frame", frame);
		imshow("inversed", inversed);
		if (waitKey(33) == 27)
			break;
	}
}
void video_in()
{
	VideoCapture cap("stopwatch.avi");
	if (!cap.isOpened()) {
		cerr << "Video open failed" << endl;
		return;
	}
	cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame height: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;
	cout << "Frame count: " << cvRound(cap.get(CAP_PROP_FRAME_COUNT)) << endl;
	double fps = cap.get(CAP_PROP_FPS);
	cout << "FPS: " << fps << endl;
	int delay = cvRound(1000 / fps);     //기본단위가 밀레세크 라 1000을 곱해서 세크로 바꿔준다.
	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		inversed = ~frame;
		imshow("frame", frame);
		imshow("inversed", inversed);
		if (waitKey(100) == 27)
			break;
	}
	destroyAllWindows();
}
void camera_in()
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
	}
	//cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	//cout << "Frame height: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;
	Mat frame, inverse;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		inverse = ~frame;
		//	outputVideo << inversed;
		imshow("frame", frame);
		imshow("inversed", inverse);
		if (waitKey(10) == 27)
			break;
	}
	destroyAllWindows();
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;
void drawLines();
void drawPolys();
int main()
{
	//	drawLines();
	drawPolys();
}
void drawPolys()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));
	rectangle(img, Rect(50, 50, 100, 50), Scalar(0, 0, 255), 2);            // 앞의 두개 : 왼쪽 상단 점 , 뒤 2개 : 그 점으로부터 x,y 거리
	rectangle(img, Rect(50, 150, 100, 50), Scalar(0, 0, 128), -1);
	circle(img, Point(300, 120), 30, Scalar(255, 255, 0), -1, LINE_AA);
	circle(img, Point(300, 120), 60, Scalar(255, 0, 0), 3, LINE_AA);
	ellipse(img, Point(120, 300), Size(60, 30), 20, 0, 270, Scalar(255, 255, 0), FILLED, LINE_AA);
	ellipse(img, Point(120, 300), Size(100, 50), 20, 0, 360, Scalar(0, 255, 0), 2, LINE_AA);
	vector<Point> pts;
	pts.push_back(Point(250, 250)); pts.push_back(Point(300, 250));
	pts.push_back(Point(300, 300)); pts.push_back(Point(350, 300));
	pts.push_back(Point(350, 350)); pts.push_back(Point(250, 350));
	polylines(img, pts, true, Scalar(255, 0, 255), 2);
	imshow("img", img);
	waitKey();
	destroyAllWindows();
}
void drawLines()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));
	line(img, Point(50, 50), Point(200, 50), Scalar(0, 0, 255), 1);
	line(img, Point(50, 100), Point(200, 100), Scalar(255, 0, 255), 3);
	line(img, Point(50, 150), Point(200, 150), Scalar(255, 0, 0), 10);
	line(img, Point(250, 50), Point(350, 100), Scalar(0, 0, 255), 1, LINE_4);
	line(img, Point(250, 70), Point(350, 120), Scalar(255, 0, 255), 3, LINE_8);
	line(img, Point(250, 90), Point(350, 140), Scalar(255, 0, 0), 1, LINE_AA);
	arrowedLine(img, Point(50, 200), Point(150, 200), Scalar(0, 0, 255), 1);
	arrowedLine(img, Point(50, 250), Point(350, 250), Scalar(255, 0, 255), 1);
	arrowedLine(img, Point(50, 300), Point(350, 300), Scalar(255, 0, 0), 1, LINE_8, 0, 0.05);
	drawMarker(img, Point(50, 350), Scalar(0, 0, 255), MARKER_CROSS);
	drawMarker(img, Point(100, 350), Scalar(0, 0, 255), MARKER_TILTED_CROSS);
	drawMarker(img, Point(150, 350), Scalar(0, 0, 255), MARKER_STAR);
	drawMarker(img, Point(200, 350), Scalar(0, 0, 255), MARKER_DIAMOND);
	drawMarker(img, Point(250, 350), Scalar(0, 0, 255), MARKER_SQUARE);
	drawMarker(img, Point(300, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_UP);
	drawMarker(img, Point(350, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_DOWN);
	imshow("img", img);
	waitKey();
	destroyAllWindows();
}
#endif

#if 0

#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

void drawLines();
void drawPolys();
void drawText1();
void drawText2();

int main(void)
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);	// 로깅 메세지들 띄우지 않기
	drawLines();
	drawPolys();
	drawText1();
	drawText2();

	return 0;
}

void drawLines()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));

	line(img, Point(50, 50), Point(200, 50), Scalar(0, 0, 255));
	line(img, Point(50, 100), Point(200, 100), Scalar(255, 0, 255), 3);
	line(img, Point(50, 150), Point(200, 150), Scalar(255, 0, 0), 10);

	line(img, Point(250, 50), Point(350, 100), Scalar(0, 0, 255), 1, LINE_4);
	line(img, Point(250, 70), Point(350, 120), Scalar(255, 0, 255), 1, LINE_8);
	line(img, Point(250, 90), Point(350, 140), Scalar(255, 0, 0), 1, LINE_AA);

	arrowedLine(img, Point(50, 200), Point(150, 200), Scalar(0, 0, 255), 1);
	arrowedLine(img, Point(50, 250), Point(350, 250), Scalar(255, 0, 255), 1);
	arrowedLine(img, Point(50, 300), Point(350, 300), Scalar(255, 0, 0), 1, LINE_8, 0, 0.05);

	drawMarker(img, Point(50, 350), Scalar(0, 0, 255), MARKER_CROSS);
	drawMarker(img, Point(100, 350), Scalar(0, 0, 255), MARKER_TILTED_CROSS);
	drawMarker(img, Point(150, 350), Scalar(0, 0, 255), MARKER_STAR);
	drawMarker(img, Point(200, 350), Scalar(0, 0, 255), MARKER_DIAMOND);
	drawMarker(img, Point(250, 350), Scalar(0, 0, 255), MARKER_SQUARE);
	drawMarker(img, Point(300, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_UP);
	drawMarker(img, Point(350, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_DOWN);

	imshow("img", img);
	waitKey();

	destroyAllWindows();
}

void drawPolys()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));

	rectangle(img, Rect(50, 50, 100, 50), Scalar(0, 0, 255), 2);
	rectangle(img, Rect(50, 150, 100, 50), Scalar(0, 0, 128), -1);

	circle(img, Point(300, 120), 30, Scalar(255, 255, 0), -1, LINE_AA);
	circle(img, Point(300, 120), 60, Scalar(255, 0, 0), 3, LINE_AA);

	ellipse(img, Point(120, 300), Size(60, 30), 20, 0, 270, Scalar(255, 255, 0), FILLED, LINE_AA);
	ellipse(img, Point(120, 300), Size(100, 50), 20, 0, 360, Scalar(0, 255, 0), 2, LINE_AA);

	vector<Point> pts;
	pts.push_back(Point(250, 250)); pts.push_back(Point(300, 250));
	pts.push_back(Point(300, 300)); pts.push_back(Point(350, 300));
	pts.push_back(Point(350, 350)); pts.push_back(Point(250, 350));
	polylines(img, pts, true, Scalar(255, 0, 255), 2);

	imshow("img", img);
	waitKey();

	destroyAllWindows();
}

void drawText1()
{
	Mat img(500, 800, CV_8UC3, Scalar(255, 255, 255));

	putText(img, "FONT_HERSHEY_SIMPLEX", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
	putText(img, "FONT_HERSHEY_PLAIN", Point(20, 100), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
	putText(img, "FONT_HERSHEY_DUPLEX", Point(20, 150), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));
	putText(img, "FONT_HERSHEY_COMPLEX", Point(20, 200), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0));
	putText(img, "FONT_HERSHEY_TRIPLEX", Point(20, 250), FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 0));
	putText(img, "FONT_HERSHEY_COMPLEX_SMALL", Point(20, 300), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 0, 0));
	putText(img, "FONT_HERSHEY_SCRIPT_SIMPLEX", Point(20, 350), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255, 0, 255));
	putText(img, "FONT_HERSHEY_SCRIPT_COMPLEX", Point(20, 400), FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(255, 0, 255));
	putText(img, "FONT_HERSHEY_COMPLEX | FONT_ITALIC", Point(20, 450), FONT_HERSHEY_COMPLEX | FONT_ITALIC, 1, Scalar(255, 0, 0));

	imshow("img", img);
	waitKey();

	destroyAllWindows();
}

void drawText2()
{
	Mat img(200, 640, CV_8UC3, Scalar(255, 255, 255));

	const String text = "Hello, OpenCV";
	int fontFace = FONT_HERSHEY_TRIPLEX;
	double fontScale = 2.0;
	int thickness = 1;

	Size sizeText = getTextSize(text, fontFace, fontScale, thickness, 0);
	Size sizeImg = img.size();

	Point org((sizeImg.width - sizeText.width) / 2, (sizeImg.height + sizeText.height) / 2);
	putText(img, text, org, fontFace, fontScale, Scalar(255, 0, 0), thickness);
	rectangle(img, org, org + Point(sizeText.width, -sizeText.height), Scalar(0, 255, 0), 1);

	imshow("img", img);
	waitKey();

	destroyAllWindows();
}

#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{
	Mat img = imread("MS.bmp");

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("img");
	imshow("img", img);

	while (true) {
		int keycode = waitKey();

		if (keycode == 'i' || keycode == 'I') {
			img = ~img;
			imshow("img", img);
		}
		else if (keycode == 27 || keycode == 'q' || keycode == 'Q') {
			break;
		}
	}

	return 0;
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat img;
Point ptOld;
void on_mouse(int event, int x, int y, int flags, void*);

int main(void)
{
	img = imread("MS.bmp");

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("img");
	setMouseCallback("img", on_mouse);

	imshow("img", img);

	
	waitKey();
	while (true) {
		int keycode = waitKey();

		if (keycode == 'i' || keycode == 'I') {
			img = ~img;
			imshow("img", img);
		}
		else if (keycode == 27 || keycode == 'q' || keycode == 'Q') {
			break;
		}
	}
	return 0;
}

void on_mouse(int event, int x, int y, int flags, void*)
{
	switch (event) {
	case EVENT_LBUTTONDOWN:
		ptOld = Point(x, y);
		cout << "EVENT_LBUTTONDOWN: " << x << ", " << y << endl;
		break;
	case EVENT_LBUTTONUP:
		cout << "EVENT_LBUTTONUP: " << x << ", " << y << endl;
		break;
	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON) {
			line(img, ptOld, Point(x, y), Scalar(255, 255, 0), 2);
			imshow("img", img);
			ptOld = Point(x, y);
		}
		break;
	default:
		break;
	}
}
#endif

#if 0
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void on_level_change(int pos, void* userdata);

int main(void)
{
	Mat img = Mat::zeros(400, 400, CV_8UC1);

	namedWindow("image", WINDOW_NORMAL);
	createTrackbar("level", "image", 0, 16, on_level_change, (void*)&img);

	imshow("image", img);
	waitKey();
	return  0;
}
void on_level_change(int pos, void* userdata)
{
	Mat img = *(Mat*)userdata;

	img.setTo(pos * 16);
	imshow("image", img);
}
#endif

#if 0
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void writeData();
void readData();

// String filename = "mydata.xml";
String filename = "mydata.yml";
//String filename = "mydata.json";

int main(void)
{
	writeData();
	readData();

	return 0;
}

void writeData()
{
	String name = "Jane";
	int age = 10;
	Point pt1(100, 200);
	vector<int> scores = { 80, 90, 50 };
	Mat mat1 = (Mat_<float>(2, 2) << 1.0f, 1.5f, 2.0f, 3.2f);

	FileStorage fs;
	fs.open(filename, FileStorage::WRITE);

	if (!fs.isOpened()) {
		cerr << "File open failed!" << endl;
		return;
	}

	fs << "name" << name;
	fs << "age" << age;
	fs << "point" << pt1;
	fs << "scores" << scores;
	fs << "data" << mat1;

	fs.release();
}

void readData()
{
	String name;
	int age;
	Point pt1;
	vector<int> scores;
	Mat mat1;

	FileStorage fs(filename, FileStorage::READ);

	if (!fs.isOpened()) {
		cerr << "File open failed!" << endl;
		return;
	}

	fs["name"] >> name;
	fs["age"] >> age;
	fs["point"] >> pt1;
	fs["scores"] >> scores;
	fs["data"] >> mat1;

	fs.release();

	cout << "name: " << name << endl;
	cout << "age: " << age << endl;
	cout << "point: " << pt1 << endl;
	cout << "scores: " << Mat(scores).t() << endl;
	cout << "data:\n" << mat1 << endl;
}
#endif

#if 1
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void mask_setTo();
void mask_copyTo();
void time_inverse();
void useful_func();

int main(void)
{
	mask_setTo();
	mask_copyTo();
	time_inverse();
	useful_func();

	return 0;
}

void mask_setTo()
{
	Mat src = imread("lenna.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_smile.bmp", IMREAD_GRAYSCALE);

	if (src.empty() || mask.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	src.setTo(Scalar(0, 255, 255), mask);

	imshow("src", src);
	imshow("mask", mask);

	waitKey();
	destroyAllWindows();
}

void mask_copyTo()
{
	Mat src = imread("airplane.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_plane.bmp", IMREAD_GRAYSCALE);
	Mat dst = imread("field.bmp", IMREAD_COLOR);

	if (src.empty() || mask.empty() || dst.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	src.copyTo(dst, mask);

	imshow("src", src);
	imshow("dst", dst);
	imshow("mask", mask);

	waitKey();
	destroyAllWindows();
}

void time_inverse()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());

	TickMeter tm;
	tm.start();

	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			dst.at<uchar>(j, i) = 255 - src.at<uchar>(j, i);
		}
	}

	tm.stop();
	cout << "Image inverse took " << tm.getTimeMilli() << "ms." << endl;
}

void useful_func()
{
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	cout << "Sum: " << (int)sum(img)[0] << endl;
	cout << "Mean: " << (int)mean(img)[0] << endl;

	double minVal, maxVal;
	Point minPos, maxPos;
	minMaxLoc(img, &minVal, &maxVal, &minPos, &maxPos);

	cout << "minVal: " << minVal << " at " << minPos << endl;
	cout << "maxVal: " << maxVal << " at " << maxPos << endl;

	Mat src = Mat_<float>({ 1, 5 }, { -1.f, -0.5f, 0.f, 0.5f, 1.f });

	Mat dst;
	normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	cout << "src: " << src << endl;
	cout << "dst: " << dst << endl;

	cout << "cvRound(2.5): " << cvRound(2.5) << endl;
	cout << "cvRound(2.51): " << cvRound(2.51) << endl;
	cout << "cvRound(3.4999): " << cvRound(3.4999) << endl;
	cout << "cvRound(3.5): " << cvRound(3.5) << endl;
}
#endif