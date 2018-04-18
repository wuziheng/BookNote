# 169 (E) Majority Element

*Given an array of size n, find the majority element. The majority element is the element that appears **more than** `⌊ n/2 ⌋` times.*

*You may assume that the array is non-empty and the majority element always exist in the array.*



**AC1: hash table: stl map 实现，主要复习map的声明，数组[]方式调用，所有value初始化都是0, Time=O(n), space=O(k)**

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        map<int,int> s;
        for(int i=0;i<nums.size();++i){
            s[nums[i]]+=1;
            if(s[nums[i]]>nums.size()/2)
                return nums[i];
        }
    }
};
```

**AC2: Boyer-Moore Voting Algorithm:记录该某一个candidate出现的次数，majority一定会被投票出来, Time = O(n), space= O(1)**

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candidate=nums[0];
        int vote=0;
        for(auto num: nums){
            if(vote==0)
                candidate=num;
            vote += (candidate==num)?1:-1;
        }
        return candidate;
    }
};
```



# 217(E)Contains Duplicate

*Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.*



**AC1: hash table, stl map, Time = O(n), space = O(n)**

```c++
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        map<int,int> s;
        for(auto num: nums){
            if(s[num]!=1)
                s[num]++;
            else
                return true;
        }
        return false;
    }
};
```



# 189(E)Rotate Array

*Rotate an array of n elements to the right by k steps.*

*For example, with n = 7 and k = 3, the array `[1,2,3,4,5,6,7]` is rotated to `[5,6,7,1,2,3,4]`.* 



**AC1: rotate 3 time ,easy. 复习一下swap函数, k 是任意整数，无聊。 Time=O(n), Space=O(1)** 

```c++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k= k%nums.size();
        if(k==0)
            return;
        rotate_a(nums,0,nums.size()-1);
        rotate_a(nums,0,k-1);
        rotate_a(nums,k,nums.size()-1);
        return;
    }
    
    void rotate_a(vector<int>& nums,int a, int b){
        if(a==b)
            return;
        for(int i=0; i<=(b-a)/2;++i)
            swap(nums[a+i], nums[b-i]);
    }
};
```



# 88(E)Merge Sorted Array

*Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.*



AC1: 双指针，注意从空间已经在尾部申请好了,最后检查一下是否nums2是否移动完了。

```c++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i{m-1};
        int j{n-1};
        int tar{m+n-1};
      
        while(i>=0 && j>=0){
            if(nums2[j]>nums1[i])
                nums1[tar--] = nums2[j--];
            else
                nums1[tar--] = nums1[i--];
        }
      
        if(i==-1){
            for(int k=0;k<=j;++k)
                nums1[k]=nums2[k];
        }
        
        return;
    }
};
```



# 118(E)Pascal's Triangle

*Given numRows, generate the first numRows of Pascal's triangle.*



**AC1:brute-force,能过就好 Time=O(n_2), space=O(n_2)** 

```c++
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res;
        if(numRows==0)
            return res;
        else if(numRows==1){
            res.push_back(vector<int>({1}));
                return res;
        }
        else{
            res.push_back(vector<int>({1}));
            for(int i=2;i<=numRows;++i){
                vector<int> tmp{1};
                vector<int> last = res.back();
                for(int j=0;j<last.size()-1;++j)
                    tmp.push_back(last[j]+last[j+1]);
                tmp.push_back(1);
                res.push_back(tmp);
            }
        }
        return res;
    }
};
```



# 442(M)Find All Duplicates in an Array

*Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear **twice** and others appear **once**.*

*Find all the elements that appear **twice** in this array.*



**AC1:  题目数字范围特殊，所以可以用每个数字对应在数组中的位置的数字的正负来描述是否出现过该数字。注意如下两点:**

1. **数字范围和可以寻址的范围有1的差距。**
2. **遍历的时候要取abs,因为该数字有可能在之前就已经被修改为负数。**

```c++
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        for(int i=0; i< nums.size();++i){
            if(nums[abs(nums[i])-1]<0)
                res.push_back(abs(nums[i]));
            
            nums[abs(nums[i])-1] = -nums[abs(nums[i])-1];
            //cout<<nums[i]<<':'<<nums[nums[i]-1]<<endl;
        }
        return res;
    }
};
```



# 75(M)Sort Colors

*Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.*

*Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.*



**AC1: 题目指明不要用O(2n),所以最后用了三指针遍历Time=O(n), space = O(1)的方式。注意停止条件。m==h才是停止条件，因为存在m==1的前进条件是不做任何动作的。如果出现h=0,m=1，前进停止就错了。所以m==h也是需要判断的**。

```c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int l{0};
        int h{nums.size()-1};
        int m{0};
        while(m<=h){
            if(nums[m]==1){
                m++;
            }
            else if(nums[m]==0){
                if(nums[l]>0)
                    swap(nums[l++],nums[m]);
                else{
                    l++;
                    m++;
                }
            }
            else{
                if(nums[h]<2){
                    swap(nums[h--],nums[m]);
                }
                else{
                    h--;
                }
            }
            cout<<l<<' '<<m<<' '<<h<<endl;
        }
    }
};
```



# 62(M)Unique Paths

*A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).*

*The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).*

*How many possible unique paths are there?*



**AC1: DP标准的记账本的写法。记得初始化。**

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> map(m+1,vector<int>(n+1,0));
        for(int i=1;i<m+1;++i){
            for(int j=1;j<n+1;++j){
                if(i==1 && j==1)
                    map[i][j]=1;
                else
                    map[i][j]= map[i-1][j]+map[i][j-1];
            }
        }
        return map[m][n];
    }
};
```



# 90(M)Subset II

*Given a collection of integers that might contain duplicates, **nums**, return all possible subsets (the power set).*



**AC1：全排列subset 有 pow(2, nums.size()) 种,然后利用stl排序和vector.erase去掉重复的即可。其中注意是要subset,所以path在push_back的时候排序一下。区分[1,2,1,2]和[1,1,2,2]这种。**

**当然如果你在开始列举之前就sort一下，也可以避免上面的重复。**

```c++
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> res;
        res.push_back(vector<int>());
        vector<int> path;
        for(int i=0;i<nums.size();++i){
            int k = res.size();
            for(int j=0;j<k;++j){
                auto a = res[j];
                a.push_back(nums[i]);
                sort(a.begin(),a.end());
                res.push_back(a);
            }
        }
        
        sort(res.begin(),res.end());
        res.erase(unique(res.begin(),res.end()),res.end());
        return res;
    }
};
```



# 560(E) Subarray Sum Equals K

*Given an array of integers and an integer **k**, you need to find the total number of continuous subarrays whose sum equals to **k**.*

***Note:***

1. *The length of the array is in range [1, 20,000].*
2. *The range of numbers in the array is [-1000, 1000] and the range of the integer **k** is [-1e7, 1e7].*



**AC1:最後都沒有AC，因爲OJ有問題。此題陷阱很多，想一個Time=O(n_2)，space=O(n)的穩過的方法如下，邏輯簡單，實現容易。 一開始想用雙指針，後來發現可以是負數，還是不太好實現。**

```c++
class soulution{
	int subarraySum(vector<int>& nums, int k){
        if(nums.size()==0 || k==0)
            return 0;
        
        vector<int> sums(nums.size()+1,0);
        for(int i=1;i<nums.size()+1;++i)
            sums[i] = nums[i-1]+sums[i-1];
        
        int res{0};
        for(int i=0;i<sums.size();++i){
            for(int j=i+1;j<sums.size()+1;++j){
                if(sums[j]-sums[i]==k)
                    res++;
            }
        }
        return res;
    }
};
```

