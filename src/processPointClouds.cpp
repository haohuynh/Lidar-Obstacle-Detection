// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Fill in the function to do voxel grid point reduction and region based filtering
  	typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
	pcl::VoxelGrid<PointT> vg;
  	vg.setInputCloud (cloud);
  	vg.setLeafSize (filterRes, filterRes, filterRes);
  	vg.filter (*cloud_filtered);
  
    typename pcl::PointCloud<PointT>::Ptr cloud_region(new pcl::PointCloud<PointT>);
  	pcl::CropBox<PointT> region(true);
  	region.setMin(minPoint);
  	region.setMax(maxPoint);
  	region.setInputCloud(cloud_filtered);
  	region.filter(*cloud_region);
  	
  	std::vector<int> indices;
  	pcl::CropBox<PointT> roof(true);
  	region.setMin(Eigen::Vector4f(-1.5, -1.7, -1 ,1));
  	region.setMax(Eigen::Vector4f(2.6, 1.7, -.4, 1));
  	region.setInputCloud(cloud_region);
  	region.filter(indices);
  
  	pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
  	for (auto index : indices){
    	inliers->indices.push_back(index);
    }
  
  	pcl::ExtractIndices<PointT> extract;
  	extract.setInputCloud(cloud_region);
  	extract.setIndices(inliers);
   	extract.setNegative(true);
  	extract.filter(*cloud_region);
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud_region;

}

template<typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::RansacPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol){
	std::unordered_set<int> inliersResult;
	srand(time(NULL));
		
  	while(maxIterations--){
      std::unordered_set<int> inliers;
      while (inliers.size() < 3){
      	inliers.insert(rand() % cloud->points.size());
      }
      
      auto it = inliers.begin();
      
      float x1 = cloud->points[*it].x;
      float y1 = cloud->points[*it].y;
      float z1 = cloud->points[*it].z;
      
      it++;
      float x2 = cloud->points[*it].x;
      float y2 = cloud->points[*it].y;
      float z2 = cloud->points[*it].z;
      
      it++;
      float x3 = cloud->points[*it].x;
      float y3 = cloud->points[*it].y;
      float z3 = cloud->points[*it].z;
      
      float A = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1);
      float B = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1);
      float C = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1);
      float D = - A*x1  - B*y1 - C*z1;
      
      for (int i = 0; i < cloud->points.size(); i++){
      	if (inliers.count(i) > 0){
        	continue;
        }
        
        if((fabs(A*cloud->points[i].x + B*cloud->points[i].y + C*cloud->points[i].z + D)/sqrt(A*A + B*B + C*C)) <= distanceTol){
          inliers.insert(i);
        }
      }
      
      if (inliers.size() > inliersResult.size()){
      	inliersResult = inliers;
      }
    }

  	return inliersResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	
  	std::unordered_set<int> inliers = RansacPlane(cloud, maxIterations, distanceThreshold);
  	typename pcl::PointCloud<PointT>::Ptr  cloudInliers(new pcl::PointCloud<PointT>());
	typename pcl::PointCloud<PointT>::Ptr cloudOutliers(new pcl::PointCloud<PointT>());
  
  	for(int index = 0; index < cloud->points.size(); index++){
		PointT point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}
     
  	if (cloudOutliers->points.size () == 0){
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      
    }
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloudOutliers,cloudInliers);
    return segResult;
}

template<typename PointT>
void ProcessPointClouds<PointT>::euclideanCluster(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol){
	processed[index] = true;
	cluster.push_back(index);
  
  	std::vector<int> nearest = tree->search(points[index], distanceTol);
  
  	for (auto i : nearest){
    	if (!processed[i]){
          	euclideanCluster(i, points, cluster, processed, tree, distanceTol);
        }
    }	
  
}

template<typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol, int minSize, int maxSize){

	// Fill out this function to return list of indices for each cluster

	std::vector<std::vector<int>> clusters;
 	std::vector<bool> processed(points.size(), false);
  	
  	for (int i = 0; i < points.size(); i++){
    	if (processed[i]){
          continue;
        }
      	
      	std::vector<int> cluster;
      	euclideanCluster(i, points, cluster, processed, tree, distanceTol);
        
        int size = cluster.size();
        if ((size >= minSize) && (size <= maxSize)){
        	clusters.push_back(cluster);
        }
      		
      
    }
  
	return clusters;

}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize){

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> cloud_clusters;

    // Perform euclidean clustering to group detected obstacles
    KdTree* tree = new KdTree;
    std::vector<std::vector<float>> points;
    std::vector<std::vector<int>> index_clusters;
    
    int id = 0;
    for (auto& point : cloud->points){
    	std::vector<float> pt = {point.x, point.y, point.z};
    	points.push_back(pt);
        tree->insert(pt, id++);
    }
    
    index_clusters = euclideanCluster(points, tree, clusterTolerance, minSize, maxSize);
    
    for(auto& indices : index_clusters){
    	typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        
  		for(auto& index: indices){
  			cloud_cluster->points.push_back(cloud->points[index]);
        }    
        
  		cloud_cluster->width = cloud_cluster->points.size();
      	cloud_cluster->height = 1;
      	cloud_cluster->is_dense = true;
     	
      	cloud_clusters.push_back(cloud_cluster);
  	}
    
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << cloud_clusters.size() << " clusters" << std::endl;

    return cloud_clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}