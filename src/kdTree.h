/* \author Aaron Brown */
// Quiz on implementing kd tree. Modified for 3D environment.

// Structure to represent node of kd tree
struct Node
{
	std::vector<float> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<float> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}
		
  	void insert(Node* & node, uint depth, std::vector<float> point, int id){
		if (node == NULL){
        	node = new Node(point, id);
        }else{
          	uint index = depth % 3; // 3-Dimensions
          
          	if (point[index] < node->point[index]){
            	insert(node->left, depth + 1, point, id);
            }else{
              	insert(node->right, depth + 1, point, id);
            }
        }
	}
  
	void insert(std::vector<float> point, int id)
	{
		// Fill in this function to insert a new point into the tree
		// the function should create a new node and place correctly with in the root 
		insert(root, 0, point, id);
	}
	
  	void search(std::vector<float> target, Node* node, int depth, float distanceTol, std::vector<int>& ids)
	{
		if (node != NULL){
          
          if (((target[0] - distanceTol) <= node->point[0] && (target[0] + distanceTol) >= node->point[0]) &&
          	((target[1] - distanceTol) <= node->point[1] && (target[1] + distanceTol) >= node->point[1])){
          	float distance = sqrt((target[0] - node->point[0])*(target[0] - node->point[0]) + (target[1] - node->point[1])*(target[1] - node->point[1]));
            if (distance <= distanceTol){
            	ids.push_back(node->id);
            }
          }
          
    	  uint index = depth % 3; // 3-Dimensions
          if((target[index] - distanceTol) < node->point[index]){
              search(target, node->left, depth + 1, distanceTol, ids);
          }
          if((target[index] + distanceTol) > node->point[index]){
              search(target, node->right, depth + 1, distanceTol, ids);
          }    	
        }
            
    }
  
	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<float> target, float distanceTol)
	{
		std::vector<int> ids;
      	search (target, root, 0, distanceTol, ids);
        return ids;
	}
	

};




