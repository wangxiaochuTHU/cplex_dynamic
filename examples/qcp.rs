extern crate cplex_dynamic;
use cplex_dynamic::{
    Constraint, ConstraintType, Env, ObjectiveType, Problem, ProblemType, Variable, VariableType,
    VariableValue, WeightedVariable,
};

fn main() {
    let num_n = 6_usize;
    let dim = 3_usize;
    let env = Env::new().unwrap();

    // starting moment
    let t1 = std::time::Instant::now();
    // populate it with a problem
    let mut prob = Problem::new(&env, "miqpex1").unwrap();

    for i in 0..dim * num_n * 4 {
        let name = format!("x{:02}", i);
        let _x = prob
            .add_variable(Variable::new(
                VariableType::Continuous,
                0.0,
                -100.0,
                100.0,
                name,
            ))
            .unwrap();
    }
    let p_s = [1.0, 0.0, -1.0];
    let v_s = [0.0, 0.0, 0.0];
    let a_s = [0.0, 0.0, 0.0];
    let p_g = [8.0, 18.0, 5.0];
    let v_g = [0.0, 0.0, 5.0];
    let a_g = [0.0, 0.0, 0.0];

    let dt: f64 = 1.0;
    let mut num_con = 0;

    // init constraint
    for k in 0_usize..3 {
        // position
        let mut dummy = Constraint::new(ConstraintType::Eq, p_s[k], format!("dummy{}", num_con));
        dummy.add_wvar(WeightedVariable::new_idx(9 + k, 1.0));
        prob.add_constraint(dummy).unwrap();
        num_con += 1;

        // velocity
        let mut dummy = Constraint::new(ConstraintType::Eq, v_s[k], format!("dummy{}", num_con));
        dummy.add_wvar(WeightedVariable::new_idx(6 + k, 1.0));
        prob.add_constraint(dummy).unwrap();
        num_con += 1;

        //accelerate
        let mut dummy = Constraint::new(ConstraintType::Eq, a_s[k], format!("dummy{}", num_con));
        dummy.add_wvar(WeightedVariable::new_idx(3 + k, 2.0));
        prob.add_constraint(dummy).unwrap();
        num_con += 1;
    }

    // final constraint
    let base = (num_n - 1) * 12;
    for k in 0_usize..3 {
        // position
        let mut dummy = Constraint::new(ConstraintType::Eq, p_g[k], format!("dummy{}", num_con));
        dummy.add_wvar(WeightedVariable::new_idx(base + k, dt.powi(3)));
        dummy.add_wvar(WeightedVariable::new_idx(base + 3 + k, dt.powi(2)));
        dummy.add_wvar(WeightedVariable::new_idx(base + 6 + k, dt));
        dummy.add_wvar(WeightedVariable::new_idx(base + 9 + k, 1.0));
        prob.add_constraint(dummy).unwrap();
        num_con += 1;

        // velocity
        let mut dummy = Constraint::new(ConstraintType::Eq, v_g[k], format!("dummy{}", num_con));
        dummy.add_wvar(WeightedVariable::new_idx(base + k, 3.0 * dt.powi(2)));
        dummy.add_wvar(WeightedVariable::new_idx(base + 3 + k, 2.0 * dt));
        dummy.add_wvar(WeightedVariable::new_idx(base + 6 + k, 1.0));
        prob.add_constraint(dummy).unwrap();
        num_con += 1;

        //accelerate
        let mut dummy = Constraint::new(ConstraintType::Eq, a_g[k], format!("dummy{}", num_con));
        dummy.add_wvar(WeightedVariable::new_idx(base + k, 6.0 * dt));
        dummy.add_wvar(WeightedVariable::new_idx(base + 3 + k, 2.0));
        prob.add_constraint(dummy).unwrap();
        num_con += 1;
    }

    // continuity

    for i in 0..num_n - 1 {
        let base = i * 12;
        for k in 0_usize..dim {
            // position
            let mut dummy = Constraint::new(ConstraintType::Eq, 0.0, format!("dummy{}", num_con));
            dummy.add_wvar(WeightedVariable::new_idx(base + k, dt.powi(3)));
            dummy.add_wvar(WeightedVariable::new_idx(base + 3 + k, dt.powi(2)));
            dummy.add_wvar(WeightedVariable::new_idx(base + 6 + k, dt));
            dummy.add_wvar(WeightedVariable::new_idx(base + 9 + k, 1.0));
            dummy.add_wvar(WeightedVariable::new_idx(base + 12 + 9 + k, -1.0));
            prob.add_constraint(dummy).unwrap();
            num_con += 1;

            // velocity
            let mut dummy = Constraint::new(ConstraintType::Eq, 0.0, format!("dummy{}", num_con));
            dummy.add_wvar(WeightedVariable::new_idx(base + k, 3.0 * dt.powi(2)));
            dummy.add_wvar(WeightedVariable::new_idx(base + 3 + k, 2.0 * dt));
            dummy.add_wvar(WeightedVariable::new_idx(base + 6 + k, 1.0));
            dummy.add_wvar(WeightedVariable::new_idx(base + 12 + 6 + k, -1.0));
            prob.add_constraint(dummy).unwrap();
            num_con += 1;

            //accelerate
            let mut dummy =
                Constraint::new(ConstraintType::Eq, a_g[k], format!("dummy{}", num_con));
            dummy.add_wvar(WeightedVariable::new_idx(base + k, 6.0 * dt));
            dummy.add_wvar(WeightedVariable::new_idx(base + 3 + k, 2.0));
            dummy.add_wvar(WeightedVariable::new_idx(base + 12 + 3 + k, -2.0));
            prob.add_constraint(dummy).unwrap();
            num_con += 1;
        }
    }

    // maximize the objective
    let mut qp_vars: Vec<usize> = Vec::new();
    let dummy = Constraint::new(ConstraintType::Eq, 0.0, "obj");
    for i in 0..num_n {
        let base = i * 12;
        for k in 0..dim {
            qp_vars.push(base + k);
            // dummy.add_wvar(WeightedVariable::new_idx(base + k, 0.0));
        }
    }
    prob.set_qp_objective(ObjectiveType::Minimize, dummy, qp_vars.clone(), dt / 36.0)
        .unwrap();

    // solve the problem
    let sol = prob.solve(ProblemType::MixedInteger).unwrap();

    let t2 = std::time::Instant::now();

    println!("time cost {:?}", t2 - t1);

    println!("{:.4?}", sol);
    let sol_vars = sol.variables;
    let mut f = 0.0_f64;
    for idx in qp_vars.into_iter() {
        let value = sol_vars[idx];
        if let VariableValue::Continuous(v) = value {
            f += v.powi(2);
        }
    }
    f = f * dt / 36.0 / 2.0;
    println!("f = {}", f);
}
