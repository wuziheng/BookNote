# 454(M)4Sum II

*Given four lists A, B, C, D of integer values, compute how many tuples `(i, j, k, l)` there are such that `A[i] + B[j] + C[k] + D[l]` is zero.*

*To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.*



**AC1: Time=O(n2), space=O(n2),拿两组数字之和做hash，和two sum没区别。**

**PS:这里我们测试样例N=500,SPACE=25000,我们可以明显看到map和unordered_map在速度上的差距。**

```c++
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        int res{0};
        
        map<int,int> s;
        
        for(int i=0;i<A.size(); ++i){
            for(int j=0; j<B.size(); ++j){
                s[A[i]+B[j]]+=1;
            }
        }
        
        for(int i=0;i< C.size();++i){
            for(int j=0; j< D.size();++j){
                if(s.find(-C[i]-D[j])!=s.end())
                    res+=s[-C[i]-D[j]];
            }
        }
        
        return res;
        
    }
        
};
```





# 378(M)Kth Smallest Element in a Sorted Matrix

*Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.*

*Note that it is the kth smallest element in the sorted order, not the kth distinct element.*



**AC1:本题是一个巧妙的二分搜索，我们的目标是搜索矩阵中的一个值。难度在于我们无法像在sort list中那样通过index得到某个值的排序信息。我们的方法在于通过二分的方法得到值，然后利用矩阵已经排序的信息快速得到该值在整个矩阵中的排序位置信息。**

1. **通过二分的方法迭代获取比较值mid.**
2. **从上往下遍历获取所有比mid小的元素的个数，由于我们的矩阵每一行都是排序过的，可以直接用stl的upper_bound方法获取在该行中比mid小的元素个数。这个的stl实现也是二分法。**
3. **巧妙的设置迭代的累加策略保证返回值一定是矩阵中的元素。**

```c++
class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n{matrix.size()};
        int l{matrix[0][0]};
        int h{matrix[n-1][n-1]};
        
        while(l<h){
            int mid = (l+h)/2;
            int num=0;
            for(int i=0;i<n;++i){
                    int pos = upper_bound(matrix[i].begin(),matrix[i].end(),mid) - matrix[i].begin();
                    num+=pos;
                    cout<<pos<<' '<<num<<endl;
            }
            if(num<k)
                l = mid+1;
            else
                h = mid;
        }
        return l;
        
    }
};
```



# 718(M)Maximum Length of Repeated Subarray

*Given two integer arrays `A` and `B`, return the maximum length of an subarray that appears in both arrays.*



**AC1:直接打表，存下所有proposal,暴力搜索即可。实际上可以在获取proposal表格后优化处理，把time=O(mn)**

```c++
class Solution {
public:
    int findLength(vector<int>& A, vector<int>& B) {
        int m = A.size();
        int n = B.size();

        vector<vector<int>> dif(m,vector<int>(n,0));

        vector<pair<int,int>> proposal;
        for(int i=m-1;i>=0;--i){
            for(int j=n-1;j>=0;--j){
                dif[i][j] = A[i]-B[j];
                if(dif[i][j]==0){
                    if(i == m-1 || j == n-1 || dif[i+1][j+1]!=0)
                        proposal.push_back(make_pair(i,j));
                }
            }
        }

        int res=0;
        for(auto pro:proposal){
            int l = pro.first;
            int r = pro.second;
            while(l>0 && r>0){
                if(dif[l-1][r-1]==0){
                    l--;
                    r--;
                }
                else
                    break;
            }

            int tmp = pro.first-l+1;
            res = max(tmp,res);

        }
        return res;
    }
};
```



**AC2: DP 打一个１维度的表格就可以了，time=O(mn)**

```c++
class Solution {
public:
    int findLength(vector<int>& a, vector<int>& b) {
        int m = a.size(), n = b.size();
        if (!m || !n) return 0;
        vector<int> dp(n + 1);
        int res = 0;
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                res = max(res, dp[j] = a[i] == b[j] ? 1 + dp[j + 1] : 0);
            }
        }
        return res;
    }
};
```



# 50(M)Pow(x,n)

*Implement [pow(x, n)](http://www.cplusplus.com/reference/valarray/pow/).*



**AC1:递归＋二分法 **

```c++
class Solution {
public:
       double myPow(double x, int n) {
        if (n==0) return 1;
        double t = pow(x,n/2);
        if (n%2) {
            return n<0 ? 1/x*t*t : x*t*t;
        } else {
            return t*t;
        }
    }
};
```



**Memory Limit:本来觉得递归会产生重复计算的需求，但是上面的方法很好的规避了这个问题。所以下面的写法就是显得很蠢～看看而已。**

```c++
class Solution{    
public:
		double myPow(double x, int n) {
        if(n==0)
            return 1.0;
        else if(n<0)
            return 1.0/myPow(x,-n);
        else{
            pow.push_back(1.0);
            pow.push_back(x);

            int i=1;
            while(i<n){
                pow.push_back(pow.back()*pow.back());
                i*=2;
            }
            vector<int> Bx = getB(n);
            double res = 1.0;
            for(int i=0;i<Bx.size();++i)
                res*=Bx[i]==1?pow[i+1]:1.0;

            return res;
        
        }

    }
    
    private:
        vector<double> pow{};
            
    vector<int> getB(int n){
        vector<int> res{};
        while(n>0){
            int i = n%2;
            res.push_back(i);
            n/=2;
        }
        return res;
    }
｝；
```



# 658(M)Find K Closest Elements

*Given a sorted array, two integers `k` and `x`, find the `k` closest elements to `x` in the array.  The result should also be sorted in ascending order.*
*If there is a tie,  the smaller elements are always preferred.*

1. *The value k is positive and will always be smaller than the length of the sorted array.*
2.  Length of the given array is positive and will not exceed 104
3.  *Absolute value of elements in the array and x will not exceed 104*



**AC1:技不如人，要多学习用lower_bound,upper_bound**

```c++
class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        int index = std::lower_bound(arr.begin(), arr.end(), x) - arr.begin();
        int i = index - 1, j = index;                                    
        while(k--) (i<0 || (j<arr.size() && abs(arr[i] - x) > abs(arr[j] - x) ))? j++: i--;
        return vector<int>(arr.begin() + i + 1, arr.begin() + j );
    }
};
```

