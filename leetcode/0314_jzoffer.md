# 二维数组中的查找

*在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。*

**AC1:从右上角开始往左下遍历。**

```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int m = array.size();
        if(m==0) return false;
        int n = array[0].size();
        if(n==0) return false;
        
        int i = 0;
        int j = n-1;
        
        while(j>=0 && i<m){
            if(array[i][j]>target)
                j--;
            else if(array[i][j]<target)
                i++;
            else
                return true;
        }
        return false;
    }
};
```



# 替换空格

*请实现一个函数，将一个字符串中的空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。*

**AC1: Sb题目，什么年代了还用char*，问题是明显的char数组长度足够大了这种东西也不说一声**

```c++
class Solution {
public:
    void replaceSpace(char *str,int length) {
        string s;
        int j=0;
        while(j<length){
            while(j<length && str[j]!=32){
                s+=str[j++];
            }
            j+=1;
            if(j>=length)
                break;
            else
                s+="%20";
        }

        for(int i=0;i<s.size();++i)
            str[i]=s[i];
        
    }
};
```



# 从尾到头打印列表

*输入一个链表，从尾到头打印链表每个节点的值*

AC1:遍历，push到vector,调stl reverse;

```c++
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> res;
        if(head ==nullptr) return res;
        while(head!=nullptr){
            res.push_back(head->val);
            head = head->next;
        }
        reverse(res.begin(),res.end());
        return res;
    }
};
```



# 重建二叉树

*输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回*

**ac1:找到中序中的root,分两边递归即可。牛客网傻逼OJ**

```c++
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
            if(!pre.size()) return nullptr;
            if(pre.size()==1){
                TreeNode* a = new TreeNode(pre[0]);
                return a;
            }
            TreeNode* a = new TreeNode(pre[0]);
            
            int i=0;
            while(vin[i]!=pre[0]) i++;
            vector<int> lpre(pre.begin()+1,pre.begin()+1+i);
            vector<int> lvin(vin.begin(),vin.begin()+i);
            vector<int> rpre(pre.begin()+1+i,pre.end());
            vector<int> rvin(vin.begin()+1+i,vin.end());
            a->left = reConstructBinaryTree(lpre,lvin);
            a->right = reConstructBinaryTree(rpre,rvin);
        
            return a;
    }
};
```



# 两个栈实现队列

*用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。*

**AC1:两个stack倒一下就好了。**

```c++
class Solution
{
public:
    void push(int node) {
        while(!stack2.empty()){
            stack1.push(stack2.top());
            stack2.pop();
        }
        stack1.push(node);
        while(!stack1.empty()){
            stack2.push(stack1.top());
            stack1.pop();
        }
    }

    int pop() {
        int a = stack2.top();
        stack2.pop();
        return a;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```





# 旋转数组最小数字

*把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。  输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。*
*例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。  NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。*

**AC1:遍历就好了，二分都懒得用。**

```c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.size()==0) return 0;
        if(rotateArray.size()==1) return rotateArray[0];
        int i=0;
        while(i<rotateArray.size()-2){
            if(rotateArray[i+1]<rotateArray[i])
                return rotateArray[i+1];
            else
                i++;
        }
        return min(rotateArray[0],rotateArray.back());
    }
};
```



# 斐波那契数列

*大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。n<=39*

**AC1:递归打表，这个是最好的方法（可惜傻逼牛客不让过）**

```c++
class Solution {
public:
    int Fibonacci(int n) {
        if(n==1) return 0;
        if(n==2) return 1;

        if(m.find(n-1)== m.end())
            m[n-1] = Fibonacci(n-1);
        if(m.find(n-2)== m.end())
            m[n-2] = Fibonacci(n-2);

        return m[n-1]+m[n-2];
    }

private:
    map<int,int> m;
};
```

**AC2:暴力解（傻逼牛客过了）**

```c++
class Solution {
public:
    int Fibonacci(int n) {
        if(n==0) return 0;
        if(n==1) return 1;
        int i=1;
        int a = 0;
        int b = 1;
        int tmp = b;
        while(i++<n){
            tmp = b;
            b = a+b;
            a = tmp;
        }
        return b;
    }
};
```



# 跳台阶

*一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。*

**AC1:DP打表即可**

```c++
class Solution {
public:
    int jumpFloor(int number) {
        if(number==1) return 1;
        if(number==2) return 2;
        if(m.find(number-1)==m.end())
            m[number-1] = jumpFloor(number-1);
        if(m.find(number-2)==m.end())
            m[number-2] = jumpFloor(number-2);
        return m[number-1]+m[number-2];
    }
    
private:
    map<int,int> m;
};
```



# 变态跳台阶

*一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。*

**AC1:考虑每一个节点，青蛙是否停留，中间有number-1个节点，所以有pow(2,number-1)种方法。**

```c++
class Solution {
public:
    int jumpFloorII(int number) {
        if(number==0) return 1;
        if(number==1) return 1;
        else return pow(2,number-1);
        
    }
};
```



# 矩阵覆盖

*我们可以用2x1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2x1的小矩形无重叠地覆盖一个2xn的大矩形，总共有多少种方法？*

**AC1:牛客网傻逼**

```c++
class Solution {
public:
    int rectCover(int number) {
        if(number<1) return 0;
        if(number==1) return 1;
        if(number==2) return 2;
        return rectCover(number-1)+rectCover(number-2);    
    }
};
```



# 二进制中１的个数

*输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示*

**AC1:（n-1）&n 可以取到最右边的１**

```c++
class Solution {
public:
     int  NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            ++count;
            n = (n - 1) & n;
        }
        return count;
     }
};
```

**AC2:判断正负数＋移位，简单直观**

```c++
class Solution {
public:
     int  NumberOf1(int n) {
         int count{0};
         if(n<0){
             n = n&0x7FFFFFFF;
             ++count;
         }
         while(n!=0){
             count+=n&1;
             n=n>>1;
         }
         return count;
     }
};
```



# 数值的整数次方

*给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。*

**AC1:首先判断正负，修改base.　exponent为正后，用二分的思想递归即可。注意对奇数的处理**

```c++
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent==0) return 1.0;
        if(exponent<0){
            exponent = -exponent;
            base = 1/base;
        }
        if(exponent%2==0)
            return Power(base,exponent/2)*Power(base,exponent/2);
        else
            return Power(base,exponent/2)*Power(base,exponent/2)*base;
        
    }
};
```



# 调整数组顺序使奇数位于偶数前面

*输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。*

**AC1:傻逼牛客网。**

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        if(array.size()<2) return;
        
        vector<int> tmp(array.size(),0);
        int j=0;
        for(auto i: array){
            if(i%2==1)
                tmp[j++]=i;
        }
        for(auto i:array){
            if(i%2==0)
                tmp[j++]=i;
        }
        
        for(int i=0;i<array.size();++i)
            array[i] = tmp[i];
    }
};
```



# 链表中倒数第k个数字

*输入一个链表，输出该链表中倒数第k个结点。*

**AC1:两个节点差Ｋ个　一起开跑**

```c++
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        
        if(pListHead==NULL||k==0)
            return NULL;
        ListNode*pTail=pListHead,*pHead=pListHead;
        for(int i=1;i<k;++i)
        {
            if(pHead->next!=NULL)
                pHead=pHead->next;
            else
                return NULL;
        }
        while(pHead->next!=NULL)
        {
            pHead=pHead->next;
            pTail=pTail->next;
        }
        return pTail;
    }
};
```



# 反转链表

*输入一个链表，反转链表后，输出链表的所有元素。*

**AC1:经典的递归，注意递归后怎么接回去。**

```c++
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if(pHead==nullptr||pHead->next==nullptr) return pHead;
        ListNode* newl = ReverseList(pHead->next);
        pHead->next->next = pHead;
        pHead->next = nullptr;
        return newl;
    }
};
```



# 合并两个排序的链表

*输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。*

AC1:

