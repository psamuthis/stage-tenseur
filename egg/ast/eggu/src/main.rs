use std::{fs::File, io::{self, BufRead, BufReader}};

use egg::*;

define_language! {
    enum MathLanguage {
        "div" = Division([Id; 2]),
        "mult" = Multiplication([Id; 2]),
        "add" = Addition([Id; 2]),
        "sub" = Substraction([Id; 2]),
        "weird_func" = WeirdFunc([Id; 2]),
        Symbol(String),
        Num(i32),
    }
}

fn read_file<P: AsRef<std::path::Path>>(filename: P) -> io::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = reader.lines()
        .collect::<Result<_, _>>()?;

    while lines.last().map(|l| l.trim().is_empty()).unwrap_or(false) {
        lines.pop();
    }

    Ok(lines)
}

fn main() -> io::Result<()> {
    let rules: &[Rewrite<MathLanguage, ()>] = &[
        rewrite!("add-commut"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("mul-commut"; "(mult ?a ?b)" => "(mult ?b ?a)"),
        rewrite!("mult-assoc"; "(mult ?a (mult ?b ?c))" => "(mult ?c (mult ?a ?b))"),
        rewrite!("self-div"; "(div ?x ?x)" => "1"),
        rewrite!("mult-one"; "(mult ?a 1)" => "?a"),
        rewrite!("add-zero"; "(add ?a 0)" => "?a"),
    ];

    let lines = read_file("/home/rousseau/Documents/egg/optimizer/dump.txt")?;

    for line in lines {
        let start: RecExpr<MathLanguage> = line.parse().unwrap();
        let runner = Runner::default().with_explanations_enabled().with_expr(&start).run(rules);

        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        println!("Rewrote {} to {}", start, best_expr);
    }

    Ok(())
}