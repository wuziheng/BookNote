# 167(E)Two Sum II - Input array is sorted

*Given an array of integers that is already **sorted in ascending order**, find two numbers such that they add up to a specific target number.*

*The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.*

*You may assume that each input would have exactly one solution and you may not use the same element twice.*

​	

**AC1:题目放在二分搜索里面，但时用hash table 是比较直观的做法。实际上二分搜索更加耗时**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        map<int,int> s;
        for(int i=0;i<numbers.size();++i){
            if(s.find(numbers[i])!=s.end())
                return {s[numbers[i]],i+1};
            else
                s[target-numbers[i]]=i+1;
        }
    }
};
```



**AC2: 双指针头尾搜索也是非常快的算法，空间上有优势，利用了sort的性质**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int l{0};
        int r{numbers.size()-1};
        while(l<r){
            if(numbers[l]+numbers[r]==target)
                return {l+1,r+1};
            else if(numbers[l]+numbers[r]<target)
                l++;
            else
                r--;
        }      
    }
};
```



# 230(M)Kth Smallest Element in a BST

*Given a binary search tree, write a function `kthSmallest` to find the **k**th smallest element in it.*

***Note: ***
*You may assume k is always valid, 1 ≤ k ≤ BST's total elements.*



**AC1: 先递归打个map（防止重复计算）:把每一个节点的为root的树的大小记录下来，然后利用bst的性质二分搜索，如果k>count(left)，说明在右边，否则在左边。**

```c++
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        int total=count(root);
        auto iter=root;
        int res{0};
        while(iter!=nullptr){
            if(count(iter->left)<k-1){
                k = k-count(iter->left)-1;
                iter = iter->right;
            }
            else if(count(iter->left)+order>k-1){
                iter = iter->left;
            }
            else{
                res = iter->val;
                break;
            }
        }
        return res;
    }
    
    map<TreeNode*, int> s;
    
    int count(TreeNode* a){
        if(s.find(a)!=s.end()){
            return s[a];
        }
        else{
            if(a==nullptr)
                return 0;
            else{
                s[a]=count(a->left)+count(a->right)+1;
                return count(a->left)+count(a->right)+1;
            }
        }
    }
};
```



# 74(M)Search a 2D Matrix

*Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:*

- *Integers in each row are sorted from left to right.*
- *The first integer of each row is greater than the last integer of the previous row.*



**AC1: 二分搜索的标准王者题目，一遍写对才是真本事。**

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if(matrix.size()==0)
            return false;
        if(matrix[0].size()==0)
            return false;
        
        int m{matrix.size()};
        int n{matrix[0].size()};
        
        int l{0};
        int r{m-1};
        if(l==r && matrix[l][n-1]==target)
            return true;
        while(l<r){
            int mid=l+(r-l)/2;
            if(matrix[mid][n-1]==target)
                return true;
            else if(matrix[mid][n-1]<target)
                l = mid+1;
            else
                r = mid;
        }
        
        int row{r};
        
        l=0;
        r=n-1;
        
//         if(l==r && matrix[row][l]==target)
//             return true;
        while(l<=r){
            int mid=l+(r-l)/2;
            cout<<row<<' '<<matrix[row][mid]<<endl;
            if(matrix[row][mid]==target)
                return true;
            else if(matrix[row][mid]<target)
                l = mid+1;
            else
                r = mid-1;
        }
        return false;   
    }
};
```



# 69(E)Sqrt(x)

*Implement `int sqrt(int x)`.*

*Compute and return the square root of x.*

***x** is guaranteed to be a non-negative integer.*



**AC1:二分经典题目，写了很多次都不对。**

```c++
class Solution {
public:
    int mySqrt(int x) {
        int low = 0,  high = x, mid;
        if(x<2) return x; // to avoid mid = 0
        while(low<high)
        {
            mid = (low + high)/2;
            if(x/mid >= mid) low = mid+1;
            else high = mid;
        }
        return high-1;
        
    }
};
```

