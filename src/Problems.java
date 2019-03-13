

import java.util.*;

/**
 * @Author: Berg
 * @Date: 2019年2月25日10:17:51
 */
public class Problems {
    /**
     * Problem1:
     * 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     * <p>
     * 解法：由于这个矩阵的性质，分别向右向下递增，那么从左下开始查找，如果当前数值小于目标值，则当前列都比目标值小，不考虑当前列 col++ ；
     * 反之，如果当前数值大于目标值，则当前行都比目标值大，不考虑当前行 row --；直到找到目标值或者出边界
     * <p>
     * 注意：行号和列号不要反了
     */
    public boolean find(int target, int[][] array) {
        int x = array.length - 1;
        int y = array[0].length - 1;
        for (int i = 0; i <= x; i++) {
            for (int j = y; j >= 0; j--) {
                if (target == array[i][j]) {
                    return true;
                } else if (target > array[i][j]) {
                    break;
                }
            }
        }
        return false;
    }

    /**
     * Problem2:
     * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     * <p>
     * 解法：从后往前，先计算需要多少空间，然后从后往前移动，则每个字符只为移动一次，这样效率更高一点。
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null)
            return null;
        int cnt = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ')
                cnt++;
        }
        int oldIndex = str.length() - 1;
        int repIndex = str.length() + cnt * 2 - 1;
        str.setLength(repIndex + 1);//使str的长度扩大到转换成%20之后的长度,防止下标越界
        for (; oldIndex < repIndex && oldIndex >= 0; oldIndex--) {
            if (str.charAt(oldIndex) != ' ')
                str.setCharAt(repIndex--, str.charAt(oldIndex));
            else {
                str.setCharAt(repIndex--, '0');
                str.setCharAt(repIndex--, '2');
                str.setCharAt(repIndex--, '%');
            }
        }
        return str.toString();
    }

    /**
     * Problem3:
     * 输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
     * <p>
     * 解法：递归，或借用栈
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode == null)
            return null;
        Stack<Integer> stack = new Stack<>();
        while (listNode != null) {
            stack.push(listNode.val);
            listNode = listNode.next;
        }
        ArrayList<Integer> arrayList = new ArrayList<>();
        while (!stack.isEmpty())
            arrayList.add(stack.pop());
        return arrayList;
    }

    //递归
    ArrayList<Integer> arrayRecursion = new ArrayList<>();

    public ArrayList<Integer> printListFromTailToHeadByRecursion(ListNode listNode) {
        if (listNode != null) {
            if (listNode.next != null)
                printListFromTailToHeadByRecursion(listNode.next);
            arrayRecursion.add(listNode.val);
        }
        return arrayRecursion;
    }

    /**
     * Problem4:
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     * <p>
     * 解法：用递归来做，注意开始和结尾下标的转换方程
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        TreeNode root = reConstructBTree(pre, 0, pre.length - 1, in, 0, in.length - 1);
        return root;
    }

    private TreeNode reConstructBTree(int[] pre, int sPre, int ePre, int[] in, int sIn, int eIn) {
        if (sPre > ePre || sIn > eIn)
            return null;
        TreeNode root = new TreeNode(pre[sPre]);
        for (int i = sIn; i <= eIn; i++) {
            if (pre[sPre] == in[i]) {
                root.left = reConstructBTree(pre, sPre + 1, i - sIn + sPre, in, sIn, i - 1);
                root.right = reConstructBTree(pre, i - sIn + sPre + 1, ePre, in, i + 1, eIn);
            }
        }
        return root;
    }

    /**
     * Problem5:
     * 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
     * <p>
     * 解法：
     */
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (!stack2.isEmpty())
            return stack2.pop();
        if (stack1.isEmpty())
            System.out.println("栈空");
        while (!stack1.isEmpty())
            stack2.push(stack1.pop());
        return stack2.pop();
    }

    /**
     * Problem6:
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。
     * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
     * <p>
     * 解法：非减排序意味着可能等于，查询单个数值一般用二分法
     * 如果待查询的范围最后只剩两个数，那么mid一定会指向下标靠前的数字
     */
    public int minNumberInRotateArray(int[] array) {
        if (array.length == 0)
            return 0;
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (array[right] > array[mid])
                right = mid;
            else if (array[mid] == array[right])//值相同情况下
                right--;
            else
                left = mid + 1;
        }
        return array[left];
    }

    /**
     * Problem7:
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法？
     * （先后次序不同算不同的结果）
     * <p>
     * 解法：动态规划，f(1)=1，f(2)=2。f(n) = f(n-1) + f(n-2)
     */
    static int JumpFloor(int target) {
        if (target <= 0)
            return -1;
        else if (target == 1)
            return 1;
        else if (target == 2)
            return 2;
        else
            return JumpFloor(target - 1) + JumpFloor(target - 2);

    }

    /**
     * Problem8:
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     * <p>
     * 解法：设第一次跳了3阶，则剩下的可以表示为从第三阶开始跳的次数，f(n-3)。
     * 以此类推，第一次跳了n次时，f(n) = f(n-1)+f(n-2)+···+f(n-n) = f(0)+f(1)+···+f(n-1)
     * f(n-1) = f(0)+f(1)+···+f((n-1)-1)
     * 上下相减得f(n) = 2f(n-1)
     */
    static int JumpFloorII(int target) {
        if (target <= 0)
            return 1;
        else if (target == 1)
            return 1;
        else
            return JumpFloorII(target - 1) * 2;

    }

    /**
     * Problem9：
     * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     * <p>
     * 解法：这种求多少种方法的一般用动态规划,分析第一块竖排，和第一二块横排情况
     */
    public int RectCover(int target) {
        if (target <= 0)
            return 0;
        else if (target == 1)
            return 1;
        else if (target == 2)
            return 2;
        else
            return RectCover(target - 1) + RectCover(target - 2);
    }

    /**
     * Problem10:
     * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     * <p>
     * 解法：如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，
     * 原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。
     */
    public int NumberOf1(int n) {
        int cnt = 0;
        while (n != 0) {
            ++cnt;
            n = n & (n - 1);
        }
        return cnt;
    }

    /**
     * Problem11:
     * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     * <p>
     * 解法：将次方分为奇数偶数考虑
     * 注意：次方可能是负数和0。数字可能是0.
     */
    public double Power(double base, int exponent) {
        if (exponent < 0)
            return 1 / powerWithUnsignedExponent(base, -exponent);
        return powerWithUnsignedExponent(base, exponent);
    }

    private double powerWithUnsignedExponent(double base, int exponent) {
        if (exponent == 0)
            return 1;
        if (base == 0)
            return 0;
        if (exponent == 1)
            return base;
        double res = Power(base, exponent >> 1);//除2
        res *= res;
        if (exponent % 2 == 1)
            res *= base;
        return res;
    }

    /**
     * Problem12:
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
     * 所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     * <p>
     * 解法：1）可以先找偶数，后找奇数进行交换。2）空间换时间
     */
    public void reOrderArray(int[] array) {
        if (array == null)
            return;
        int[] rArray = new int[array.length];
        int index = 0;
        int rIndex = 0;
        for (int num : array) {
            if ((num & 1) == 1)//相当于num%2==1
                array[index++] = num;
            else
                rArray[rIndex++] = num;
        }
        for (int i = 0; i < rIndex; i++) {
            array[index + i] = rArray[i];
        }
    }

    /**
     * Problem12:
     * 输入一个链表，输出该链表中倒数第k个结点。
     * <p>
     * 解法：双指针，跟着第一个指针，距离k
     * 注意：这个是没有头结点的，倒数第一个处理，和倒数最后最length的节点
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        ListNode kNode = head;
        while (k-- > 0) {
            if (head == null)
                return null;
            head = head.next;
        }
        while (head != null) {
            head = head.next;
            kNode = kNode.next;
        }
        return kNode;
    }

    /**
     * Problem13:
     * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     * <p>
     * 解法：用一个新的链表串起来
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null)
            return list2;
        if (list2 == null)
            return list1;
        ListNode curr = new ListNode(0);
        ListNode head = curr;
        while (list1 != null && list2 != null) {
            if (list1.val > list2.val) {
                curr.next = list2;
                list2 = list2.next;
            } else {
                curr.next = list1;
                list1 = list1.next;
            }
            curr = curr.next;
        }
        if (list1 == null)
            curr.next = list2;
        if (list2 == null)
            curr.next = list1;
        return head.next;
    }

    /**
     * Problem14:
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     * <p>
     * 解法：递归
     * 注意：子树的意思是只要包含了一个结点，就得包含这个结点下的所有节点.
     * 子结构的意思是包含了一个结点，可以只取左子树或者右子树，或者都不取。
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean res = false;
        if (root1 != null && root2 != null) {
            if (root1.val == root2.val)
                res = isSubtree(root1, root2);
            if (!res)
                res = isSubtree(root1.left, root2) || isSubtree(root1.right, root2);
        }
        return res;
    }

    private boolean isSubtree(TreeNode root1, TreeNode root2) {
        if (root2 == null) return true;
        if (root1 == null) return false;
        if (root1.val == root2.val) {
            return isSubtree(root1.left, root2.left) && isSubtree(root1.right, root2.right);
        } else
            return false;
    }

    /**
     * Problem15:
     * 操作给定的二叉树，将其变换为源二叉树的镜像。
     * <p>
     * 解法：递归,注意是每次交换是左右子树，不是简单的值交换
     * 使用栈
     */
    public void Mirror(TreeNode root) {
        if (root == null)
            return;
        if (root.left == null && root.right == null)
            return;
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        if (root.left != null)
            Mirror(root.left);
        if (root.right != null)
            Mirror(root.right);
    }

    //非递归
    public void noRecurrsionMirror(TreeNode root) {
        if (root == null)
            return;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (node.left == null && node.right == null)
                continue;
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
            if (node.left != null)
                stack.push(node.left);
            if (node.right != null)
                stack.push(node.right);
        }
    }

    /**
     * Problem16:
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
     * 例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
     * <p>
     * 解法：想象成一圈一圈打印，从外到内，注意考虑最后可能只有一行或一列或一个数
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (matrix == null || matrix.length <= 0 || matrix[0].length <= 0)
            return arrayList;
        int rows = matrix.length;
        int cols = matrix[0].length;
        for (int start = 0; start * 2 < rows && start * 2 < cols; start++) {//圈循环
            int endX = cols - start;
            int endY = rows - start;
            for (int i = start; i < endX; i++)
                arrayList.add(matrix[start][i]);
            if (endY - 1 > start)//至少两行
                for (int i = start + 1; i < endY; i++)
                    arrayList.add(matrix[i][endX - 1]);
            if (start < endX - 1 && start < endY - 1)//条件：至少两行两列
                for (int i = endX - 2; i >= start; i--)
                    arrayList.add(matrix[endY - 1][i]);
            if (start < endX - 1 && start < endY - 2)//条件：最少三行两列
                for (int i = endY - 2; i > start; i--)
                    arrayList.add(matrix[i][start]);
        }
        return arrayList;
    }

    /**
     * Problem17:
     * 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
     * <p>
     * 解法：借用辅助栈存储min的大小，自定义了栈结构
     */
    class Solution {
        private Stack<Integer> minStack = new Stack<Integer>();
        private Stack<Integer> stack = new Stack<Integer>();

        public void push(int node) {
            stack.push(node);
            if (minStack.isEmpty() || node < minStack.peek()) {
                minStack.push(node);
            } else {
                minStack.push(minStack.peek());
            }

        }

        public void pop() {
            assert !minStack.isEmpty() : "空的";
            stack.pop();
            minStack.pop();
        }

        public int top() {
            assert !stack.isEmpty() : "空的";
            return stack.peek();
        }

        public int min() {
            assert !minStack.isEmpty() : "空的";//assert[boolean 表达式 : 错误表达式 （日志）]
            return minStack.peek();
        }
    }

    /**
     * Problem18:
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
     * （注意：这两个序列的长度是相等的）
     * <p>
     * 解法:用一个辅助栈模拟出站，遇到相等就出栈
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null || pushA.length == 0 || popA.length == 0)
            return false;
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            while (j < popA.length && stack.peek() == popA[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }


    /**
     * Problem19:
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     * <p>
     * 解法：层序遍历,借助队列
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (root == null)
            return arrayList;
        Queue<TreeNode> queue = new LinkedList<>();//队列两种实现类，LinkedList，以及PriorityQueue（这个需要制定元素排序方式，弹出时取出最小元素）
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            arrayList.add(node.val);
            if (node.left != null)
                queue.add(node.left);
            if (node.right != null)
                queue.add(node.right);
        }
        return arrayList;
    }

    /**
     * Problem20:
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。
     * 假设输入的数组的任意两个数字都  互不相同。
     * <p>
     * 解法：分治法，找住二叉查找树的特点：左子树<根<=右子树
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length == 0)
            return false;
        if (sequence.length == 1)
            return true;
        return verifySubTree(sequence, 0, sequence.length - 1);
    }

    private boolean verifySubTree(int[] sequence, int start, int end) {
        if (start >= end)
            return true;
        int i = start;
        while (sequence[i] < sequence[end])
            ++i;
        for (int j = i; j < end; j++) {
            if (sequence[j] < sequence[end])
                return false;
        }
        return verifySubTree(sequence, start, i - 1) && verifySubTree(sequence, i, end - 1);
    }

    /**
     * Problem21:
     * 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。
     * （注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
     */
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null)
            return null;
        RandomListNode cur = pHead;
        RandomListNode tmp = new RandomListNode(0);
        //复制链表，新节点插入老节点后
        while (cur != null) {
            tmp = new RandomListNode(cur.label);
            tmp.next = cur.next;
            cur.next = tmp;
            cur = tmp.next;
        }
        //复制random,注意nullz值！！！
        cur = pHead;
        while (cur != null) {
            cur.next.random = cur.random == null ? null : cur.random.next;
            cur = cur.next.next;
        }
        //拆分
        cur = pHead;
        tmp = pHead.next;
        RandomListNode nHead = pHead.next;
        while (cur != null) {
            cur.next = cur.next.next;
            tmp.next = tmp.next == null ? null : tmp.next.next;
            cur = cur.next;
            tmp = tmp.next;
        }
        return nHead;
    }

    /**
     * Problem22:
     * 题目描述
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
     * 要求不能创建任何新的结点，只能调整树中结点指针的指向。
     * <p>
     * 解法：中序遍历顺序,头节点先找到特殊处理，之后处理剩下的
     */
    private TreeNode head = null;
    private TreeNode pre = null;

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null)
            return null;
        convertHelper(pRootOfTree);
        return head;
    }

    private void convertHelper(TreeNode node) {
        if (node == null)
            return;
        convertHelper(node.left);
        if (head == null) {
            head = node;
            pre = node;
        } else {
            pre.right = node;
            node.left = pre;
            pre = node;
        }
        convertHelper(node.right);
    }

    /**
     * Problem23:
     * 输入一个字符串,按字典序打印出该字符串中字符的所有排列。
     * 例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
     * <p>
     * 解法：回溯法,1）将字符串分解为两部分，第一个字符以及剩下的所有字符，然后求剩下字符的排列。
     * 2）将第一个字符与之后的字符逐一交换。
     */
    public ArrayList<String> Permutation(String str) {
        if (str == null)
            return null;
        List<String> res = new ArrayList<>();
        PermutationHelper(str.toCharArray(), 0, res);
        Collections.sort(res);
        return (ArrayList<String>) res;

    }

    private void PermutationHelper(char[] str, int begin, List<String> res) {
        if (begin == str.length - 1) {//已经移动到数组最后了
            if (!res.contains(new String(str))) {
                res.add(new String(str));
                return;
            }
        } else {
            for (int i = 0; i < str.length; i++) {
                swap(str, begin, i);
                PermutationHelper(str, begin + 1, res);
                swap(str, i, begin);//交换回来
            }
        }

    }

    private void swap(char[] str, int i, int j) {
        if (i != j) {
            char t = str[i];
            str[i] = str[j];
            str[j] = t;
        }
    }

    /**
     * Problem24:
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
     * <p>
     * 解法：打擂算法~
     * 如果有符合条件的数字，则它出现的次数比其他所有数字出现的次数和还要多。
     * 在遍历数组时保存两个值：一是数组中一个数字，一是次数。遍历下一个数字时，若它与之前保存的数字相同，则次数加1，否则次数减1；
     * 若次数为0，则保存下一个数字，并将次数置为1。遍历结束后，所保存的数字即为所求。然后再判断它是否符合条件即可。
     * <p>
     * 注意：还可以利用快排的思想，找到排序后位置在n/2的数字，该数字一定是目标
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        if (array == null || array.length == 0)
            return 0;
        int no = array[0];
        int cnt = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] == no) {
                cnt++;
            } else {
                if (--cnt <= 0) {
                    cnt = 1;
                    no = array[i];
                }
            }
        }
        cnt = 0;
        for (int i = 0; i < array.length; i++) {//确认下
            if (array[i] == no)
                cnt++;
        }
        return cnt * 2 > array.length ? no : 0;
    }

    /**
     * Problem25:
     * 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     * <p>
     * 解法：用java的priorityQueue，使用堆实现的，默认
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if (input == null || input.length == 0 || k <= 0 || k > input.length)
            return res;
        PriorityQueue<Integer> queue = new PriorityQueue<>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);//降序
            }
        });

        for (int i = 0; i < input.length; i++) {
            if (queue.size() != k) {
                queue.offer(input[i]);
            } else if (input[i] < queue.peek()) {
                queue.poll();
                queue.offer(input[i]);
            }
        }
        while (!queue.isEmpty())
            res.add(queue.poll());
        return res;
    }

    public static ArrayList<Integer> GetLeastNumbers_SolutionByHeap(int[] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if (input == null || input.length == 0 || k <= 0 || k > input.length)
            return res;
        int[] heap = new int[k];
        for (int i = 0; i < k; i++) {
            heap[i] = input[i];
        }
        //1.构建da顶堆
        for (int i = k / 2 - 1; i >= 0; i--) {//调整各个非叶子节点,直到调整整个结构
            adjustHeap(heap, i, heap.length);
        }
        //2.调整堆结构+交换堆顶元素与末尾元素，这样就从0到len由大到小排列
        /*for (int j = heap.length - 1; j > 0; j--) {
            swap(heap, 0, j);//交换堆顶与最后一个元素
            adjustHeap(heap, 0, j);//重新调整堆
        }*/
        for (int m = k; m < input.length; m++) {//从第k个元素开始分别与最大堆的最大值做比较，如果比最大值小，则替换并调整堆。
            if (input[m] < heap[0]) {
                heap[0] = input[m];
                adjustHeap(heap, 0, heap.length);
            }
        }
        for (int i = 0; i < heap.length; i++) {
            res.add(heap[i]);
        }
        return res;
    }

    private static void adjustHeap(int[] heap, int i, int len) {
        int tmp = heap[i];
        for (int k = i * 2 + 1; k < len; k = k * 2 + 1) {//i*2+1为下一左子树,遍历i节点所有子树
            if (k + 1 < len && heap[k] < heap[k + 1])//寻找最小孩子节点
                k++;
            if (heap[k] > tmp) {//如果子节点da于父节点，将子节点赋值给父节点（不交换
                heap[i] = heap[k];
                i = k;//锚定空位置
            } else
                break;
        }
        heap[i] = tmp;//将tmp放到最终位置
    }

    private static void  swap(int[] array, int i, int j) {
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }

    public static void main(String[] args) {
        int[] a = new int[8];
        a[0] = 4;
        a[1] = 5;
        a[2] = 1;
        a[3] = 6;
        a[4] = 2;
        a[5] = 7;
        a[6] = 3;
        a[7] = 8;
        GetLeastNumbers_SolutionByHeap(a,4);
    }
}




























