#include "LSQ.h"
#include <iostream>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <vector>
#include <time.h>
#include "MatrixReaderWriter.h"

using namespace cv;
using namespace std;

void FitPlaneRANSAC(
	MatrixReaderWriter& mrw,
	vector<int>& inliers,
	Mat& plane,
	double threshold,
	double confidence_,
	int iteration_number);

void FitPlaneLSQ(MatrixReaderWriter& mrw,
	vector<int>& inliers,
	Mat& plane);

size_t GetIterationNumber(
	const double& inlierRatio_,
	const double& confidence_,
	const size_t& sampleSize_);

int main()
{
	MatrixReaderWriter* mrw;
	mrw = new MatrixReaderWriter("D:/source/repos/LSQ/out/build/street.xyz");

	srand(time(NULL));

	vector<int> inliers;
	Mat bestPlane;

	// RANSAC to find the plane parameters and the inliers
	FitPlaneRANSAC(
		*mrw,
		inliers, // Output: the indices of the inliers
		bestPlane, // Output: the parameters of the found 2D line
		0.08, // The inlier-outlier threshold
		0.99, // The confidence required in the results
		1000); // The number of iterations

	MatrixReaderWriter* mrw2;
	mrw2 = mrw;
	for (const auto& inlieridx : inliers)
	{
		mrw2->data[mrw2->columnNum * inlieridx + 3] = 255;
	}
	mrw2->save("D:/source/repos/LSQ/out/build/street2.xyz");
	const double& a1 = bestPlane.at<double>(0);
	const double& b1 = bestPlane.at<double>(1);
	const double& c1 = bestPlane.at<double>(2);
	const double& d1 = bestPlane.at<double>(3);

	printf("The plane's coefficients /RANSAC/: a = %f, b = %f, c = %f, d = %f \n", a1, b1, c1, d1);

	// Calculate the error on the RANSAC plane
	double averageError1 = 0.0;
	for (const auto& inlierIdx : inliers)
	{
		Point3d p;
		p.x = mrw->data[mrw->columnNum * inlierIdx];
		p.y = mrw->data[mrw->columnNum * inlierIdx + 1];
		p.z = mrw->data[mrw->columnNum * inlierIdx + 2];
		double distance = abs(a1 * p.x + b1 * p.y + c1 * p.z + d1);
		averageError1 += distance;
	}
	averageError1 /= inliers.size();

	FitPlaneLSQ(*mrw,
		inliers,
		bestPlane);
	 const double& a2 = bestPlane.at<double>(0);
	 const double& b2 = bestPlane.at<double>(1);
	 const double& c2 = bestPlane.at<double>(2);
	 const double& d2 = bestPlane.at<double>(3);

	 printf("The plane's coefficients /LSQ/: a = %f, b = %f, c = %f, d = %f \n", a2, b2, c2, d2);

	 // Calculate the error on the Least Squares Fitting Plane
	 double averageError2 = 0.0;
	 for (const auto& inlierIdx : inliers)
	 {
		 Point3d p;
		 p.x = mrw->data[mrw->columnNum * inlierIdx];
		 p.y = mrw->data[mrw->columnNum * inlierIdx + 1];
		 p.z = mrw->data[mrw->columnNum * inlierIdx + 2];
		 double distance = abs(a2 * p.x + b2 * p.y + c2 * p.z + d2);
		 averageError2 += distance;
	 }
	 averageError2 /= inliers.size();

	 printf("Avg. RANSAC error = %f px\n", averageError1);
	 printf("Avg. LSQ error = %f px\n", averageError2);

	return 0;
}

size_t GetIterationNumber(
	const double& inlierRatio_,
	const double& confidence_,
	const size_t& sampleSize_)
{
	double a =
		log(1.0 - confidence_);
	double b =
		log(1.0 - std::pow(inlierRatio_, sampleSize_));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<size_t>::max();

	return a / b;
}

// Apply RANSAC to fit points to a plane
void FitPlaneRANSAC(
	MatrixReaderWriter& mrw,
	vector<int>& inliers_,
	Mat& plane_,
	double threshold_,
	double confidence_,
	int maximum_iteration_number_)
{
	// The current number of iterations
	int iterationNumber = 0;
	// The number of inliers of the current best model
	int bestInlierNumber = 0;
	// The indices of the inliers of the current best model
	vector<int> bestInliers, inliers;
	int inputSize = mrw.columnNum * mrw.rowNum;
	bestInliers.reserve(mrw.rowNum);
	inliers.reserve(mrw.rowNum);
	// The parameters of the best plane
	Mat bestPlane(4, 1, CV_64F);
	// Helpers to draw the plane if needed
	Point3d bestPt1, bestPt2, bestPt3;
	// The current sample
	std::vector<int> sample(3);

	size_t maximumIterations = maximum_iteration_number_;

	// RANSAC:
	// 1. Select a minimal sample, i.e., in this case, 3 random points.
	// 2. Fit a plane to the points.
	// 3. Count the number of inliers, i.e., the points closer than the threshold.
	// 4. Store the inlier number and the plane parameters if it is better than the previous best. 

	while (iterationNumber++ < maximumIterations)
	{
		// 1. Select a minimal sample, i.e., in this case, 2 random points.
		for (size_t sampleIdx = 0; sampleIdx < 3; ++sampleIdx)
		{
			do
			{
				// Generate a random index between [0, pointNumber]
				sample[sampleIdx] =
					round((mrw.rowNum - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				// If the first point is selected we don't have to check if
				// that particular index had already been selected.
				if (sampleIdx == 0)
					break;

				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand. 
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;
				if (sampleIdx == 2 &&
					sample[1] != sample[2] &&
					sample[0] != sample[2])
					break;
			} while (true);
		}
		// 2. Fit a plane to the points.
		Point3d p1; // First point selected
		p1.x = mrw.data[mrw.columnNum * sample[0]];
		p1.y = mrw.data[mrw.columnNum * sample[0] + 1];
		p1.z = mrw.data[mrw.columnNum * sample[0] + 2];
		Point3d p2; // Second point selected		
		p2.x = mrw.data[mrw.columnNum * sample[1]];
		p2.y = mrw.data[mrw.columnNum * sample[1] + 1];
		p2.z = mrw.data[mrw.columnNum * sample[2] + 2];
		Point3d p3; // Third point selected		
		p3.x = mrw.data[mrw.columnNum * sample[2]];
		p3.y = mrw.data[mrw.columnNum * sample[2] + 1];
		p3.z = mrw.data[mrw.columnNum * sample[2] + 2];

		// These two vectors are in the plane
		Point3d v1 = p3 - p1;
		Point3d v2 = p2 - p1;

		// Cross product
		Point3d cp = v1.cross(v2);
		cp = cp / cv::norm(cp); // Making it unit length 
								// --> If the normal vector is normalized (unit length), then the constant term of the plane equation, d becomes the distance from the origin.

		double a = cp.x;
		double b = cp.y;
		double c = cp.z;
		double d = cp.ddot(p3);


		// - Distance of a plane and a point
		// - Plane's implicit equations: a * x + b * y + c * z + d = 0
		// - a, b, c, d - parameters of the line
		// - x, y, z - coordinates of a point on the plane
		// - Distance(plane, point) = | a * x + b * y + c * z + d | / sqrt(a * a + b * b + c * c)
		// - If ||n||_2 = 1 then sqrt(a * a + b * b  + c * c) = 1 and I don't have do the division.

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < mrw.rowNum; ++pointIdx)
		{
			Point3d point; // iterating through the points 1 by 1
			point.x = mrw.data[mrw.columnNum * pointIdx];
			point.y = mrw.data[mrw.columnNum * pointIdx + 1];
			point.z = mrw.data[mrw.columnNum * pointIdx + 2];

			const double distance =
				abs(a * point.x + b * point.y + c * point.z + d);

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);
			}
		}

		// 4. Store the inlier number and the plane parameters if it is better than the previous best. 
		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestPlane.at<double>(0) = a;
			bestPlane.at<double>(1) = b;
			bestPlane.at<double>(2) = c;
			bestPlane.at<double>(3) = d;

			// Update the maximum iteration number
			maximumIterations = GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(mrw.rowNum),
				confidence_,
				3);

			printf("Inlier number = %d\tMax iterations = %d\n", bestInliers.size(), maximumIterations);
		}
	}

	inliers_ = bestInliers;
	plane_ = bestPlane;
}

// Apply Least-Squares line fitting (PCL).
void FitPlaneLSQ(MatrixReaderWriter& mrw,
	vector<int>& inliers,
	Mat& plane)
{
	vector<Point3d> normalizedPoints;
	normalizedPoints.reserve(inliers.size());

	// Calculating the mass point of the points
	Point3d masspoint(0, 0, 0);

	for (const auto& inlierIdx : inliers)
	{
		Point3d p;
		p.x = mrw.data[mrw.columnNum * inlierIdx];
		p.y = mrw.data[mrw.columnNum * inlierIdx + 1];
		p.z = mrw.data[mrw.columnNum * inlierIdx + 2];
		masspoint += p;
		normalizedPoints.emplace_back(p);
	}
	masspoint = masspoint * (1.0 / inliers.size());

	// Move the point cloud to have the origin in their mass point
	for (auto& point : normalizedPoints)
		point -= masspoint;

	// Calculating the average distance from the origin
	double averageDistance = 0.0;
	for (auto& point : normalizedPoints)
	{
		averageDistance += cv::norm(point);
		// norm(point) = sqrt(point.x * point.x + point.y * point.y)
	}

	averageDistance /= normalizedPoints.size();
	const double ratio = sqrt(2) / averageDistance;

	// Making the average distance to be sqrt(2)
	for (auto& point : normalizedPoints)
		point *= ratio;

	// Now, we should solve the equation.
	cv::Mat A(normalizedPoints.size(), 3, CV_64F);

	// Building the coefficient matrix
	for (size_t pointIdx = 0; pointIdx < normalizedPoints.size(); ++pointIdx)
	{
		const size_t& rowIdx = pointIdx;

		A.at<double>(rowIdx, 0) = normalizedPoints[pointIdx].x;
		A.at<double>(rowIdx, 1) = normalizedPoints[pointIdx].y;
		A.at<double>(rowIdx, 2) = normalizedPoints[pointIdx].z;
	}

	cv::Mat evals, evecs;
	cv::eigen(A.t() * A, evals, evecs);
	const cv::Mat& normal = evecs.row(2);
	const double& a = normal.at<double>(0),
		& b = normal.at<double>(1), 
		& c = normal.at<double>(2);
	Point3d normalP(a, b, c);
	const double d = -normalP.ddot(masspoint);
	
	plane.at<double>(0) = a;
	plane.at<double>(1) = b;
	plane.at<double>(2) = c;
	plane.at<double>(3) = d;
}