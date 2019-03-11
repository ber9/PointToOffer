
public class Test {
    public static void main(String[] args) {
        ListNode head = new ListNode(0);
        head.next = new ListNode(1);
        ListNode pre = head;
        head = head.next;
        System.out.println(pre.val);
    }
}
