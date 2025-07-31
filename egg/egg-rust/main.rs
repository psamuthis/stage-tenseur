use egg::*;

define_language! {
    enum MatrixLanguage {
        "krao"   = KhatriRao([Id; 2]),
        "hdmr"   = Hadamard([Id; 2]),
        "kron"   = Kronecker([Id; 2]),
        "outer"  = Outer([Id; 2]),
        "*"      = Mul([Id; 2]),
        "+"      = Add([Id; 2]),
        "T"      = Transpose(Id),
        "pinv"   = PInverse(Id),
        Symbol(String),
    }
}
 
fn main() {

    let rules: &[Rewrite<MatrixLanguage, ()>] = &[
        rewrite!("tranposed-kron-distr"; "(T (kron A B))" => "(kron (T A) (T B))"),
        rewrite!("pinv-kron-distr"; "(pinv (kron A B))" => "(kron (pinv A) (pinv B))"),
        rewrite!("mixed-product"; "(hdmr (krao A B) (krao C D))" => "(krao (hdmr A C) (hdmr B D))"),
        rewrite!("krao-crossprod"; "(* (T (krao A B)) (krao A B) )" => "(kron (* (T A) A) (* (T B) B))"),
        rewrite!("mixed-hadamard"; "(outer (kron A B) (kron C D))" => "(kron (outer A C) (outer B D))"),
    ];


    //let mut egraph: EGraph<MatrixLanguage, ()> = Default::default();

    //let B = egraph.add(MatrixLanguage::Symbol("B".into()));
    //let C = egraph.add(MatrixLanguage::Symbol("C".into()));
    //let X = egraph.add(MatrixLanguage::Symbol("X".into()));
    //let BT = egraph.add(MatrixLanguage::Transpose(B));
    //let CT = egraph.add(MatrixLanguage::Transpose(C));
    
    ////1. B %0% C
    //let krao = egraph.add(MatrixLanguage::KhatriRao([B, C]));

    ////2. pinv(CTC h*h BTB)
    //let CTC = egraph.add(MatrixLanguage::Mul([CT, C]));
    //let BTB = egraph.add(MatrixLanguage::Mul([BT, B]));
    //let h = egraph.add(MatrixLanguage::Hadamard([CTC, BTB]));
    //let p = egraph.add(MatrixLanguage::PInverse(h));

    ////1 * 2
    //let prod = egraph.add(MatrixLanguage::Mul([krao, p]));
    //let final_prod = egraph.add(MatrixLanguage::Mul([X, prod]));


    //egraph.rebuild();
    //let mttkrp_pattern: Pattern<MatrixLanguage> = "(* ?A ?B)".parse().unwrap();
    //let matches = mttkrp_pattern.search(&egraph);

    //for m in matches {
        //println!("{:?}", m);

    //}
    //println!("{:?}", egraph.dump());

    let start = "(T (kron A B))".parse().unwrap();
    let runner = Runner::default().with_explanations_enabled().with_expr(&start).run(rules);

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

    println!("Original {}", start);
    println!("{}", best_cost);
    println!("Transformed {}", best_expr);
}