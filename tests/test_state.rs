use scorpion::orbit::State;

#[test]
fn test_state_clone() {
    let s = State { a: 7e6, e: 0.01, i: 1.5, O: 2.0, o: 1.0, t: 0.5 };
    let s2 = s.clone();
    assert_eq!(s.a, s2.a);
    assert_eq!(s.e, s2.e);
    assert_eq!(s.i, s2.i);
    assert_eq!(s.O, s2.O);
    assert_eq!(s.o, s2.o);
    assert_eq!(s.t, s2.t);
}
