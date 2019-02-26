import java.util.ArrayList;
import java.util.Stack;

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
            return JumpFloorII(target - 1)*2;

    }

    /**
     * Problem9
     *
     */


    public static void main(String[] args) {
        System.out.println(JumpFloorII(5));

    }
}

















