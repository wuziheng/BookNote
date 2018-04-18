# TOPK

优先队列（最大堆实现）



```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        
        vector<int> res;
        if(input.size() < k || k <= 0) return res;
        
        priority_queue<int,vector<int>, less<int>> q;
        for(auto i : input){
            if(q.size()<k)
                q.push(i);
            else{
                if(i<q.top()){
                    q.push(i);
                    q.pop();
                }
            }
        }
        while(!q.empty()){
            res.push_back(q.top());
            q.pop();
        }
        return res;
    }
};
```



# 从上往下打印二叉树

*从上往下打印出二叉树的每个节点，同层节点从左至右打印*

**AC1: queue实现的bfs**

```c++
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        vector<int> res;
        if(root==nullptr)
            return res;
        queue<TreeNode* > q;
        q.push(root);
        while(!q.empty()){
            int s = q.size();
            for(int i=0;i<s;++i){
                res.push_back(q.front()->val);
                if(q.front()->left!=nullptr)
                    q.push(q.front()->left);
                if(q.front()->right!=nullptr)
                    q.push(q.front()->right);
                q.pop();
            }
        }
        return res;
    }
};
```





# 二叉树中和为某一值的路径

*输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。*



AC1: dps(注意引用符号不要打掉了)

```c++
class Solution {
public:
    vector<vector<int>> FindPath(TreeNode* root,int expectNumber) {
        vector<vector<int>> res;
        if(root==nullptr) return res;
        vector<int> path;
        dfs(root,expectNumber,path,res);
        return res;
    }
    
    void dfs(TreeNode* root, int expect, vector<int>& path, vector<vector<int>>& res){
        if(root==nullptr)
            return;
        if(root->val==expect && root->left==nullptr && root->right==nullptr){
            path.push_back(root->val);
            res.push_back(path);
            path.pop_back();
            return;
        }
        path.push_back(root->val);
        expect -= root->val;
        dfs(root->left, expect, path, res);
        dfs(root->right, expect, path, res);
        expect += root->val;
        path.pop_back();
        return;
    }
};
```





# 数组中出现次数超过一半的数字

*数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。*

**AC1: hashtable 稳健法（还有一种投票的方法Ｏ（１）的空间，但需要确定数字是存在的）。**

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        map<int,int> s;
        int res(0);
        for(auto i: numbers)
            s[i]++;
        for(auto it = s.begin();it!=s.end();++it){
            if(it->second>numbers.size()/2){
                res = it->first;
                break;
            }
        }
        return res;
    }
};
```



# 把数组排成最小的数

*输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。*

**AC1:写一个cmp，把int转vector之后的vector按题目需求排序，注意cmp函数里面Ａ和Ｂ长度可能不一样，要特殊处理一下**

```c++
using namespace std;
bool cmp(vector<int> A, vector<int> B){
    reverse(A.begin(),A.end());
    reverse(B.begin(),B.end());

    if(A.size()>B.size()){
        for(int i=0;i<A.size();i++) {
            int bi = i % B.size();
            if (A[i] != B[bi])
                return A[i] < B[bi];
        }
    }
    else{
        for(int i=0;i<B.size();i++) {
            int ai = i % A.size();
            if (A[ai] != B[i])
                return A[ai] < B[i];
        }
    }
    return A[0]<B[0];
}

class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        string res;
        if(numbers.size()==0) return res;

        vector<vector<int>> s;
        for(auto i:numbers){
            vector<int> tmp;
            while(i>0){
                tmp.push_back(i%10);
                i/=10;
            }
            s.push_back(tmp);
        }
        sort(s.begin(),s.end(),cmp);

        for(auto a : s){
            reverse(a.begin(),a.end());
            for(auto i : a)
                res+=(i+'0');

        }
        return res;
    }
};
```





# Ugly Number

*把只包含因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。*

**AC1：所有的丑数都是由之前的丑数产生的，而且每一个丑数都会产生新的丑数（后一句话很关键，自然数的奇妙性质）。**

```c++
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if (index < 7)return index;
        vector<int> res(index);
        res[0] = 1;
        int t2 = 0, t3 = 0, t5 = 0, i;
        for (i = 1; i < index; ++i)
        {
            res[i] = min(res[t2] * 2, min(res[t3] * 3, res[t5] * 5));
            if (res[i] == res[t2] * 2)t2++;
            if (res[i] == res[t3] * 3)t3++;
            if (res[i] == res[t5] * 5)t5++;
        }
        return res[index - 1];
    }
};
```





# 按之字形顺序打印二叉树

*请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。*

AC1: BFS+REVERSE可以解决

```c++
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        vector<vector<int>> res;
        if(pRoot==nullptr) return res;
        queue<TreeNode*> s;
        
        s.push(pRoot);
        int even = 0;
        while(!s.empty()){
            int size = s.size();
            vector<int> tmp;
            for(int i=0;i<size;++i){
                tmp.push_back(s.front()->val);
                if(s.front()->left!=nullptr)
                    s.push(s.front()->left);
                if(s.front()->right!=nullptr)
                    s.push(s.front()->right);
                s.pop();
            }
            if(even%2==1)
                reverse(tmp.begin(),tmp.end());
            res.push_back(tmp);
            even++;
        }
        return res;
    }
};
```





# 对称的二叉树

*请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。*

**AC1:先用bfs打印左右字树，然后取出来每一层进行对比。注意：打印的时候要把树补充成完全二叉树，所以还需要提前获得字树的深度。**

```c++
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        if(pRoot == nullptr || (pRoot->left==nullptr && pRoot->right==nullptr)) return true;
        vector<vector<int>> left = printTree(pRoot->left);
        vector<vector<int>> right = printTree(pRoot->right);
        if(left.size()!=right.size()) return false;
        for(int i=0;i<left.size();++i){
            vector<int> a = left[i];
            vector<int> b = right[i];
            if(a.size()!=b.size()) return false;
            reverse(b.begin(),b.end());
            for(int j=0;j<a.size();++j)
                if(a[j]!=b[j])
                    return false;
        }
        return true;
    }
    
    vector<vector<int>> printTree(TreeNode* root){
        vector<vector<int>> res;
        if(root==nullptr) return res;
        
        queue<TreeNode*> s;
        s.push(root);
        
        TreeNode zero = TreeNode(0);zhuanyi
        int i = 0;
        int k = maxDepth(root);
        
        while(i++<k){
            int size = s.size();
            vector<int> tmp;
            for(int i=0;i<size;++i){
                tmp.push_back(s.front()->val);
                
                if(s.front()->left!=nullptr)
                    s.push(s.front()->left);
                else
                    s.push(&zero);
                
                if(s.front()->right!=nullptr)
                    s.push(s.front()->right);
                else
                    s.push(&zero);
                
                s.pop();
            }
            res.push_back(tmp);
        }
        
        return res;
    }
    
    int maxDepth(TreeNode* root) {
        if(root==nullptr)
            return 0;
        else
            return max(maxDepth(root->left),maxDepth(root->right))+1;
    }
};
```





# 滑动窗口最大值

*给一个数组，返回滑动窗口内的最大值。*

**AC1:维护一个大顶堆，如果出队的数据是堆顶，则pop即可。**

```c++
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        priority_queue<int,vector<int>,less<int>> s;

        vector<int> res;
        for(int i=0;i<size;++i)
            s.push(num[i]);
        res.push_back(s.top());

        for(int i=0,j=size;j<num.size();++i,++j){
            if(num[i]==s.top())
                s.pop();
            s.push(num[j]);
            res.push_back(s.top());
        }
        return res;
    }
};
```

