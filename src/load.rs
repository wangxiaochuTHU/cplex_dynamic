// #[macro_use]
// extern crate lazy_static;

extern crate dlopen;
// #[macro_use]
// extern crate dlopen_derive;
use dlopen::wrapper::{Container, WrapperApi};
use std::path::PathBuf;

// #[macro_use]
// extern crate const_cstr;
extern crate libc;
use libc::{c_char, c_double, c_int};
use std::ffi::CString;
// extern crate dunce;

/// Used by CPLEX to represent a variable that has no upper bound.
pub const INFINITY: f64 = 1.0E+20;
pub const EPINT_ID: c_int = 2010; // identifier of integrality tolerance parameter 'EpInt'
pub const EPINT: c_double = 1e-5; // default value for 'EpInt'

pub enum CEnv {}

pub enum CProblem {}

type CInt = c_int;

lazy_static! {
    static ref CPLEX_API: dlopen::wrapper::Container<Api> = {
        let mut lib_path: Option<PathBuf> = None;
        #[cfg(target_os = "linux")]
        let pattern = "./**/cplex*.so";
        #[cfg(target_os = "windows")]
        let pattern = "./**/cplex*.dll";

        for entry in glob::glob(pattern).unwrap() {
            match entry {
                Ok(path) => {
                    lib_path = Some(path);
                    break;
                }
                Err(e) => panic!("glob error! {:?}", e),
            }
        }
        if lib_path.is_none() {
            println!("Should first put lib file (looking like `cplex***.dll` or `cplex***.so`) under the root folder or its subfolder.")
        }
        let cont: Container<Api> = unsafe { Container::load(lib_path.unwrap()) }
            .expect("Could not open library or load symbols");
        cont
    };
}

#[allow(non_snake_case)]
#[allow(dead_code)]
#[derive(WrapperApi)]
struct Api {
    CPXopenCPLEX: fn(status: *mut c_int) -> *mut CEnv,
    CPXcreateprob: fn(env: *mut CEnv, status: *mut c_int, name: *const c_char) -> *mut CProblem,
    CPXsetintparam: fn(env: *mut CEnv, param: c_int, value: c_int) -> c_int,
    CPXsetdblparam: fn(env: *mut CEnv, param: c_int, value: c_double) -> c_int,
    CPXgetintparam: fn(env: *mut CEnv, param: c_int, value: *mut c_int) -> c_int,
    CPXgetdblparam: fn(env: *mut CEnv, param: c_int, value: *mut c_double) -> c_int,
    CPXchgprobtype: fn(env: *mut CEnv, lp: *mut CProblem, ptype: c_int) -> c_int,
    // adding variables and constraints
    CPXnewcols: fn(
        env: *mut CEnv,
        lp: *mut CProblem,
        count: CInt,
        obj: *const c_double,
        lb: *const c_double,
        ub: *const c_double,
        xctype: *const c_char,
        name: *const *const c_char,
    ) -> c_int,
    CPXaddrows: fn(
        env: *mut CEnv,
        lp: *mut CProblem,
        col_count: CInt,
        row_count: CInt,
        nz_count: CInt,
        rhs: *const c_double,
        sense: *const c_char,
        rmatbeg: *const CInt,
        rmatind: *const CInt,
        rmatval: *const c_double,
        col_name: *const *const c_char,
        row_name: *const *const c_char,
    ) -> c_int,
    CPXaddlazyconstraints: fn(
        env: *mut CEnv,
        lp: *mut CProblem,
        row_count: CInt,
        nz_count: CInt,
        rhs: *const c_double,
        sense: *const c_char,
        rmatbeg: *const CInt,
        rmatind: *const CInt,
        rmatval: *const c_double,
        row_name: *const *const c_char,
    ) -> c_int,
    // querying
    CPXgetnumcols: fn(env: *const CEnv, lp: *mut CProblem) -> CInt,
    // setting objective
    CPXchgobj: fn(
        env: *mut CEnv,
        lp: *mut CProblem,
        cnt: CInt,
        indices: *const CInt,
        values: *const c_double,
    ) -> c_int,
    // changes the coefficient in the quadratic objective
    CPXchgqpcoef:
        fn(env: *mut CEnv, lp: *mut CProblem, i: CInt, j: CInt, newvalue: c_double) -> c_int,

    CPXchgobjsen: fn(env: *mut CEnv, lp: *mut CProblem, maxormin: c_int) -> c_int,
    // solving
    CPXlpopt: fn(env: *mut CEnv, lp: *mut CProblem) -> c_int,
    CPXmipopt: fn(env: *mut CEnv, lp: *mut CProblem) -> c_int,
    // getting solution
    CPXgetstat: fn(env: *mut CEnv, lp: *mut CProblem) -> c_int,
    CPXgetobjval: fn(env: *mut CEnv, lp: *mut CProblem, objval: *mut c_double) -> c_int,
    CPXgetx:
        fn(env: *mut CEnv, lp: *mut CProblem, x: *mut c_double, begin: CInt, end: CInt) -> c_int,
    CPXsolution: fn(
        env: *mut CEnv,
        lp: *mut CProblem,
        lpstat_p: *mut c_int,
        objval_p: *mut c_double,
        x: *mut c_double,
        pi: *mut c_double,
        slack: *mut c_double,
        dj: *mut c_double,
    ) -> c_int,
    // adding initial solution
    CPXaddmipstarts: fn(
        env: *mut CEnv,
        lp: *mut CProblem,
        mcnt: CInt,
        nzcnt: CInt,
        beg: *const CInt,
        varindices: *const CInt,
        values: *const c_double,
        effortlevel: *const CInt,
        mipstartname: *const *const c_char,
    ) -> c_int,
    // debugging
    CPXgeterrorstring: fn(env: *mut CEnv, errcode: c_int, buff: *mut c_char) -> *mut c_char,
    CPXwriteprob:
        fn(env: *mut CEnv, lp: *mut CProblem, fname: *const c_char, ftype: *const c_char) -> c_int,
    // freeing
    CPXcloseCPLEX: fn(env: *const *mut CEnv) -> c_int,
    CPXfreeprob: fn(env: *mut CEnv, lp: *const *mut CProblem) -> c_int,
}

fn errstr(env: *mut CEnv, errcode: c_int) -> Result<String, String> {
    unsafe {
        let mut buf = vec![0i8; 1024];
        let res = CPLEX_API.CPXgeterrorstring(env, errcode, buf.as_mut_ptr());
        if res == std::ptr::null_mut() {
            Err(format!("No error string for {}", errcode))
        } else {
            Ok(String::from_utf8(
                buf.iter()
                    .take_while(|&&i| i != 0 && i != '\n' as i8)
                    .map(|&i| i as u8)
                    .collect::<Vec<u8>>(),
            )
            .unwrap())
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ParamType {
    Integer(c_int),
    Double(c_double),
    Boolean(c_int),
}

#[derive(Copy, Clone, Debug)]
pub enum EnvParam {
    Threads(u64),
    ScreenOutput(bool),
    RelativeGap(f64),
    /// When true, set CPX_PARALLEL_DETERMINISTIC (default). When false, set
    /// CPX_PARALLEL_OPPORTUNISTIC.
    ParallelDeterministic(bool),
}

impl EnvParam {
    fn to_id(&self) -> c_int {
        use EnvParam::*;
        match self {
            &Threads(_) => 1067,
            &ScreenOutput(_) => 1035,
            &RelativeGap(_) => 2009,
            &ParallelDeterministic(_) => 1109,
        }
    }

    fn param_type(&self) -> ParamType {
        use EnvParam::*;
        use ParamType::*;
        match self {
            &Threads(t) => Integer(t as c_int),
            &ScreenOutput(b) => Boolean(b as c_int),
            &RelativeGap(g) => Double(g as c_double),
            &ParallelDeterministic(b) => Integer(if b { 1 } else { -1 }),
        }
    }
}

/// A CPLEX Environment. An `Env` is necessary to create a
/// `Problem`.
pub struct Env {
    inner: *mut CEnv,
}

impl Env {
    pub fn new() -> Result<Env, String> {
        unsafe {
            let mut status = 0;
            // println!("env = 111");
            // let env = CPXopenCPLEX(&mut status);

            let env = CPLEX_API.CPXopenCPLEX(&mut status);
            // println!("env = {:?}", env);
            if env == std::ptr::null_mut() {
                Err(format!(
                    "CPLEX returned NULL for CPXopenCPLEX (status: {})",
                    status
                ))
            } else {
                // CPXsetintparam(env, 1035, 1); //ScreenOutput
                // CPXsetintparam(env, 1056, 1); //Read_DataCheck
                Ok(Env { inner: env })
            }
        }
    }

    /// Set an environment parameter. e.g.
    ///
    /// ```
    /// use rplex::{Env, EnvParam};
    ///
    /// let mut env = Env::new().unwrap();
    /// env.set_param(EnvParam::ScreenOutput(true)).unwrap();
    /// env.set_param(EnvParam::RelativeGap(0.01)).unwrap();
    /// ```
    pub fn set_param(&mut self, p: EnvParam) -> Result<(), String> {
        unsafe {
            let status = match p.param_type() {
                ParamType::Integer(i) => CPLEX_API.CPXsetintparam(self.inner, p.to_id(), i),
                ParamType::Boolean(b) => CPLEX_API.CPXsetintparam(self.inner, p.to_id(), b),
                ParamType::Double(d) => CPLEX_API.CPXsetdblparam(self.inner, p.to_id(), d),
            };

            if status != 0 {
                return match errstr(self.inner, status) {
                    Ok(s) => Err(s),
                    Err(e) => Err(e),
                };
            } else {
                return Ok(());
            }
        }
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        unsafe {
            assert!(CPLEX_API.CPXcloseCPLEX(&self.inner) == 0);
        }
    }
}

/// A Variable in a Problem.
///
/// The general usage pattern is to create Variables outside of a
/// Problem with `var!(...)` and then add them to the Problem with
/// `prob.add_variable(...)`.
///
/// ```
/// #[macro_use]
/// extern crate rplex;
///
/// use rplex::{Env, Problem, Variable};
///
/// fn main() {
///     let env = Env::new().unwrap();
///     let mut prob = Problem::new(&env, "dummy").unwrap();
///     prob.add_variable(var!("x" -> 4.0 as Binary)).unwrap();
///     prob.add_variable(var!(0.0 <= "y" <= 100.0  -> 3.0 as Integer)).unwrap();
///     prob.add_variable(var!(0.0 <= "z" <= 4.5 -> 2.0)).unwrap();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Variable {
    index: Option<usize>,
    ty: VariableType,
    obj: f64,
    lb: f64,
    ub: f64,
    name: String,
}

impl Variable {
    pub fn new<S>(ty: VariableType, obj: f64, lb: f64, ub: f64, name: S) -> Variable
    where
        S: Into<String>,
    {
        Variable {
            index: None,
            ty: ty,
            obj: obj,
            lb: lb,
            ub: ub,
            name: name.into(),
        }
    }
}

/// Expressive creation of variables.
///
/// The general syntax is:
///
/// `var!(lower <= "name" <= upper -> objective as type)`
///
/// See `Variable` for examples.
#[macro_export]
macro_rules! var {
    ($lb:tt <= $name:tt <= $ub:tt -> $obj:tt as $vt:path) => {
        {
            use $crate::VariableType::*;
            $crate::Variable::new ($vt, $obj, $lb, $ub, $name)
        }
    };
    // continuous shorthand
    ($lb:tt <= $name:tt <= $ub:tt -> $obj:tt) => (var!($lb <= $name <= $ub -> $obj as Continuous));
    // omit either lb or ub
    ($lb:tt <= $name:tt -> $obj:tt) => (var!($lb <= $name <= INFINITY -> $obj));
    ($name:tt <= $ub:tt -> $obj:tt) => (var!(0.0 <= $name <= $ub -> $obj));
    // omit both
    ($name:tt -> $obj:tt) => (var!(0.0 <= $name -> $obj));

    // typed version
    ($lb:tt <= $name:tt -> $obj:tt as $vt:path) => (var!($lb <= $name <= INFINITY -> $obj as $vt));
    ($name:tt <= $ub:tt -> $obj:tt as $vt:path) => (var!(0.0 <= $name <= $ub -> $obj as $vt));
    ($name:tt -> $obj:tt as Binary) => (var!(0.0 <= $name <= 1.0 -> $obj as Binary));
    ($name:tt -> $obj:tt as $vt:path) => (var!(0.0 <= $name -> $obj as $vt));
}

/// A variable with weight (row) coefficient. Used in the construction
/// of `Constraint`s.
#[derive(Clone, Debug)]
pub struct WeightedVariable {
    var: usize,
    weight: f64,
}

impl WeightedVariable {
    /// Create a `WeightedVariable` from a `Variable`. Does not keep a
    /// borrowed copy of `var`.
    pub fn new_var(var: &Variable, weight: f64) -> Self {
        WeightedVariable {
            var: var.index.unwrap(),
            weight: weight,
        }
    }

    /// Create a `WeightedVariable` from a column index. This method
    /// is used by the `con!` macro, as the return value of
    /// `prob.add_variable` is a column index and thus the most common
    /// value.
    pub fn new_idx(idx: usize, weight: f64) -> Self {
        WeightedVariable {
            var: idx,
            weight: weight,
        }
    }
}

/// A Constraint (row) object for a `Problem`.
///
/// The recommended way to build these is with the `con!` macro.
///
/// ```
/// #[macro_use]
/// extern crate rplex;
///
/// use rplex::{Env, Problem, Variable};
///
/// fn main() {
///     let env = Env::new().unwrap();
///     let mut prob = Problem::new(&env, "dummy").unwrap();
///     let x = prob.add_variable(var!("x" -> 4.0 as Binary)).unwrap();
///     let y = prob.add_variable(var!(0.0 <= "y" <= 100.0  -> 3.0 as Integer)).unwrap();
///     let z = prob.add_variable(var!(0.0 <= "z" <= 4.5 -> 2.0)).unwrap();
///     prob.add_constraint(con!("dummy": 20.0 = 1.0 x + 2.0 y + 3.0 z)).unwrap();
///     prob.add_constraint(con!("dummy2": 1.0 <= (-1.0) x + 1.0 y)).unwrap();
/// }
/// ```
///
/// However, constraints can also be constructed manually from
/// `WeightedVariables`. This can be useful if your constraints don't
/// quite fit the grammar allowed by the `con!` macro. You can create
/// part of the constraint using `con!`, then augment it with
/// `add_wvar` to obtain the constraint you need.The following example
/// is identical to the above.
///
/// ```
/// #[macro_use]
/// extern crate rplex;
///
/// use rplex::{Env, Problem, Constraint, ConstraintType, WeightedVariable};
///
/// fn main() {
///     let env = Env::new().unwrap();
///     let mut prob = Problem::new(&env, "dummy").unwrap();
///     let x = prob.add_variable(var!("x" -> 4.0 as Binary)).unwrap();
///     let y = prob.add_variable(var!(0.0 <= "y" <= 100.0  -> 3.0 as Integer)).unwrap();
///     let z = prob.add_variable(var!(0.0 <= "z" <= 4.5 -> 2.0)).unwrap();
///     let mut dummy = Constraint::new(ConstraintType::Eq, 20.0, "dummy");
///     dummy.add_wvar(WeightedVariable::new_idx(x, 1.0));
///     dummy.add_wvar(WeightedVariable::new_idx(y, 2.0));
///     dummy.add_wvar(WeightedVariable::new_idx(z, 3.0));
///     prob.add_constraint(dummy).unwrap();
///     let mut dummy2 = Constraint::new(ConstraintType::GreaterThanEq, 1.0, "dummy2");
///     dummy2.add_wvar(WeightedVariable::new_idx(x, -1.0));
///     dummy2.add_wvar(WeightedVariable::new_idx(y, 1.0));
///     prob.add_constraint(dummy2).unwrap();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Constraint {
    index: Option<usize>,
    vars: Vec<WeightedVariable>,
    ty: ConstraintType,
    rhs: f64,
    name: String,
}

impl Constraint {
    pub fn new<S, F>(ty: ConstraintType, rhs: F, name: S) -> Constraint
    where
        S: Into<String>,
        F: Into<f64>,
    {
        Constraint {
            index: None,
            vars: vec![],
            ty: ty,
            rhs: rhs.into(),
            name: name.into(),
        }
    }

    /// Move a `WeightedVariable` into the Constraint.
    pub fn add_wvar(&mut self, wvar: WeightedVariable) {
        self.vars.push(wvar)
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! con_ty {
    (=) => {
        $crate::ConstraintType::Eq
    };
    (>=) => {
        $crate::ConstraintType::LessThanEq
    };
    (<=) => {
        $crate::ConstraintType::GreaterThanEq
    };
}

/// Expressive macro for writing constraints.
///
/// # Examples
/// ## Basic Form: Explicit sum of variables
/// ```
/// # #[macro_use]
/// # extern crate rplex;
/// # fn main() {
/// let x1 = 1; let x2 = 2;
/// con!("basic": 0.0 <= 1.0 x1 + (-1.0) x2);
/// # }
/// ```
/// ## Basic Form: Unweighted sum of variables
/// ```
/// # #[macro_use]
/// # extern crate rplex;
/// # fn main() {
/// let xs = vec![1, 2];
/// con!("basic sum": 0.0 >= sum (&xs));
/// # }
/// ```
/// ## Basic Form: Weighted sum of variables
/// ```
/// # #[macro_use]
/// # extern crate rplex;
/// # fn main() {
/// let xs = vec![(1, 1.0), (2, -1.0)];
/// con!("basic weighted sum": 0.0 = wsum (&xs));
/// # }
/// ```
///
/// ## Mixed Forms
/// These can be mixed at will, separated by `+`.
///
/// ```
/// # #[macro_use]
/// # extern crate rplex;
/// # fn main() {
/// let x1 = 1; let x2 = 2;
/// let ys = vec![3, 4];
/// let zs = vec![(5, 1.0), (6, -1.0)];
/// con!("mixed sum": 0.0 <= 1.0 x1 + (-1.0) x2 + sum (&ys) + wsum (&zs));
/// # }
/// ```
#[macro_export]
macro_rules! con {
    (@inner $con:ident sum $body:expr) => {
        for &var in $body {
            $con.add_wvar($crate::WeightedVariable::new_idx(var, 1.0));
        }
    };
    (@inner $con:ident wsum $body:expr) => {
        for &(var, weight) in $body {
            $con.add_wvar($crate::WeightedVariable::new_idx(var, weight));
        }
    };
    (@inner $con:ident $weight:tt $var:expr) => {
        $con.add_wvar($crate::WeightedVariable::new_idx($var, $weight));
    };
    ($name:tt : $rhs:tt $cmp:tt $c1:tt $x1:tt $(+ $c:tt $x:tt)*) => {
        {
            let mut con = $crate::Constraint::new(con_ty!($cmp), $rhs, $name);
            con!(@inner con $c1 $x1);
            $(
                con!(@inner con $c $x);
            )*
            con
        }
    };
}

/// A CPLEX Problem.
#[allow(dead_code)]
pub struct Problem<'a> {
    inner: *mut CProblem,
    /// The Environment to which the Problem belongs.
    env: &'a Env,
    /// The name of the problem.
    name: String,
    variables: Vec<Variable>,
    constraints: Vec<Constraint>,
    lazy_constraints: Vec<Constraint>,
}

/// Solution to a CPLEX Problem.
///
/// Currently, there is no way to select which variables are extracted
/// when using `prob.solve()`. I am currently unfamiliar with the C
/// API for managing variables that remain unbound in the solution,
/// and so am unsure how to represent them.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The value of the objective reached by CPLEX.
    pub objective: f64,
    /// The values bound to each variable.
    pub variables: Vec<VariableValue>,
}

#[derive(Copy, Clone, Debug)]
pub enum ObjectiveType {
    Maximize,
    Minimize,
}

#[derive(Copy, Clone, Debug)]
pub enum VariableType {
    Continuous,
    Binary,
    Integer,
    /// A variable bounded by `[lb, ub]` or equal to 0.
    SemiContinuous,
    /// An integer variable bounded by `[lb, ub]` or equal to 0.
    SemiInteger,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VariableValue {
    Continuous(f64),
    Binary(bool),
    Integer(CInt),
    SemiContinuous(f64),
    SemiInteger(CInt),
}

/// Kind of (in)equality of a Constraint.
///
/// Note that the direction of the inequality is *opposite* what one
/// might expect from the `con!` macro. This is because the right- and
/// left-hand sides are flipped in the macro.
#[derive(Copy, Clone, Debug)]
pub enum ConstraintType {
    LessThanEq,
    Eq,
    GreaterThanEq,
    /// `Ranged` is currently unimplemented.
    Ranged,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProblemType {
    Linear,
    MixedInteger,
}

impl VariableType {
    fn to_c(&self) -> c_char {
        match self {
            &VariableType::Continuous => 'C' as c_char,
            &VariableType::Binary => 'B' as c_char,
            &VariableType::Integer => 'I' as c_char,
            &VariableType::SemiContinuous => 'S' as c_char,
            &VariableType::SemiInteger => 'N' as c_char,
        }
    }
}

impl ConstraintType {
    fn to_c(&self) -> c_char {
        match self {
            &ConstraintType::LessThanEq => 'L' as c_char,
            &ConstraintType::Eq => 'E' as c_char,
            &ConstraintType::GreaterThanEq => 'G' as c_char,
            &ConstraintType::Ranged => unimplemented!(),
        }
    }
}

impl ObjectiveType {
    fn to_c(&self) -> c_int {
        match self {
            &ObjectiveType::Minimize => 1 as c_int,
            &ObjectiveType::Maximize => -1 as c_int,
        }
    }
}

impl<'a> Problem<'a> {
    pub fn new<S>(env: &'a Env, name: S) -> Result<Self, String>
    where
        S: Into<String>,
    {
        let mut status = 0;
        let name = name.into();
        let prob = CPLEX_API.CPXcreateprob(
            env.inner,
            &mut status,
            CString::new(name.as_str()).unwrap().as_ptr(),
        );
        if prob == std::ptr::null_mut() {
            Err(format!(
                "CPLEX returned NULL for CPXcreateprob ({} ({}))",
                errstr(env.inner, status).unwrap(),
                status
            ))
        } else {
            Ok(Problem {
                inner: prob,
                env: env,
                name: name,
                variables: vec![],
                constraints: vec![],
                lazy_constraints: vec![],
            })
        }
    }

    /// Add a variable to the problem. The Variable is **moved** into
    /// the problem. At this time, it is not possible to get a
    /// reference to it back.
    ///
    /// The column index for the Variable is returned.
    pub fn add_variable(&mut self, var: Variable) -> Result<usize, String> {
        let status = CPLEX_API.CPXnewcols(
            self.env.inner,
            self.inner,
            1,
            &var.obj,
            &var.lb,
            &var.ub,
            &var.ty.to_c(),
            &CString::new(var.name.as_str()).unwrap().as_ptr(),
        );

        if status != 0 {
            Err(format!(
                "Failed to add {:?} variable {} ({} ({}))",
                var.ty,
                var.name,
                errstr(self.env.inner, status).unwrap(),
                status
            ))
        } else {
            let index = CPLEX_API.CPXgetnumcols(self.env.inner, self.inner) as usize - 1;
            self.variables.push(Variable {
                index: Some(index),
                ..var
            });
            Ok(index)
        }
    }

    /// Add a constraint to the problem.
    ///
    /// The row index for the constraint is returned.
    pub fn add_constraint(&mut self, con: Constraint) -> Result<usize, String> {
        let (ind, val): (Vec<CInt>, Vec<f64>) = con
            .vars
            .iter()
            .filter(|wv| wv.weight != 0.0)
            .map(|wv| (wv.var as CInt, wv.weight))
            .unzip();
        let nz = val.len() as CInt;

        let status = CPLEX_API.CPXaddrows(
            self.env.inner,
            self.inner,
            0,
            1,
            nz,
            &con.rhs,
            &con.ty.to_c(),
            &0,
            ind.as_ptr(),
            val.as_ptr(),
            std::ptr::null(),
            &CString::new(con.name.as_str()).unwrap().as_ptr(),
        );

        if status != 0 {
            Err(format!(
                "Failed to add {:?} constraint {} ({} ({}))",
                con.ty,
                con.name,
                errstr(self.env.inner, status).unwrap(),
                status
            ))
        } else {
            let index = self.constraints.len();
            self.constraints.push(Constraint {
                index: Some(index),
                ..con
            });
            Ok(index)
        }
    }

    /// Adds a lazy constraint to the problem.
    ///
    /// Returns the index of the constraint. Unclear if this has any value whatsoever.
    pub fn add_lazy_constraint(&mut self, con: Constraint) -> Result<usize, String> {
        let (ind, val): (Vec<CInt>, Vec<f64>) = con
            .vars
            .iter()
            .filter(|wv| wv.weight != 0.0)
            .map(|wv| (wv.var as CInt, wv.weight))
            .unzip();
        let nz = val.len() as CInt;

        let status = CPLEX_API.CPXaddlazyconstraints(
            self.env.inner,
            self.inner,
            1,
            nz,
            &con.rhs,
            &con.ty.to_c(),
            &0,
            ind.as_ptr(),
            val.as_ptr(),
            &CString::new(con.name.as_str()).unwrap().as_ptr(),
        );

        if status != 0 {
            Err(format!(
                "Failed to add {:?} constraint {} ({} ({}))",
                con.ty,
                con.name,
                errstr(self.env.inner, status).unwrap(),
                status
            ))
        } else {
            let index = self.lazy_constraints.len();
            self.constraints.push(Constraint {
                index: Some(index),
                ..con
            });
            Ok(index)
        }
    }

    /// Set the objective coefficients. A Constraint object is used
    /// because it encodes a weighted sum, although it is semantically
    /// incorrect. The right-hand-side and kind of (in)equality of the
    /// Constraint are ignored.
    pub fn set_objective(&mut self, ty: ObjectiveType, con: Constraint) -> Result<(), String> {
        let (ind, val): (Vec<CInt>, Vec<f64>) = con
            .vars
            .iter()
            .map(|wv| (wv.var as CInt, wv.weight))
            .unzip();

        let status = CPLEX_API.CPXchgobj(
            self.env.inner,
            self.inner,
            con.vars.len() as CInt,
            ind.as_ptr(),
            val.as_ptr(),
        );

        if status != 0 {
            Err(format!(
                "Failed to set objective weights ({} ({}))",
                errstr(self.env.inner, status).unwrap(),
                status
            ))
        } else {
            self.set_objective_type(ty)
        }
    }

    ///
    pub fn set_qp_objective(
        &mut self,
        ty: ObjectiveType,
        con: Constraint,
        qp_var: Vec<usize>,
        beta: f64,
    ) -> Result<(), String> {
        let (ind, val): (Vec<CInt>, Vec<f64>) = con
            .vars
            .iter()
            .map(|wv| (wv.var as CInt, wv.weight))
            .unzip();

        let status = CPLEX_API.CPXchgobj(
            self.env.inner,
            self.inner,
            con.vars.len() as CInt,
            ind.as_ptr(),
            val.as_ptr(),
        );
        for i in qp_var.into_iter() {
            let _status_qp = CPLEX_API.CPXchgqpcoef(
                self.env.inner,
                self.inner,
                i as CInt,
                i as CInt,
                beta as c_double,
            );
        }

        if status != 0 {
            Err(format!(
                "Failed to set objective weights ({} ({}))",
                errstr(self.env.inner, status).unwrap(),
                status
            ))
        } else {
            self.set_objective_type(ty)
        }
    }

    /// Change the objective type. Default: `ObjectiveType::Minimize`.
    ///
    /// It is recommended to use this in conjunction with objective
    /// coefficients set by the `var!` macro rather than using
    /// `set_objective`.
    pub fn set_objective_type(&mut self, ty: ObjectiveType) -> Result<(), String> {
        let status = CPLEX_API.CPXchgobjsen(self.env.inner, self.inner, ty.to_c());
        if status != 0 {
            Err(format!(
                "Failed to set objective type to {:?} ({} ({}))",
                ty,
                errstr(self.env.inner, status).unwrap(),
                status
            ))
        } else {
            Ok(())
        }
    }

    /// Write the problem to a file named `name`. At this time, it is
    /// not possible to use a `Write` object instead, as this calls C
    /// code directly.
    pub fn write<S>(&self, name: S) -> Result<(), String>
    where
        S: Into<String>,
    {
        let status = CPLEX_API.CPXwriteprob(
            self.env.inner,
            self.inner,
            CString::new(name.into().as_str()).unwrap().as_ptr(),
            std::ptr::null(),
        );
        if status != 0 {
            return match errstr(self.env.inner, status) {
                Ok(s) => Err(s),
                Err(e) => Err(e),
            };
        } else {
            Ok(())
        }
    }

    /// Add an initial solution to the problem.
    ///
    /// `vars` is an array of indices (i.e. the result of `prob.add_variable`) and `values` are
    /// their values.
    pub fn add_initial_soln(&mut self, vars: &[usize], values: &[f64]) -> Result<(), String> {
        assert!(values.len() == vars.len());

        let vars = vars.into_iter().map(|&u| u as CInt).collect::<Vec<_>>();

        let status = CPLEX_API.CPXaddmipstarts(
            self.env.inner,
            self.inner,
            1,
            vars.len() as CInt,
            &0,
            vars.as_ptr(),
            values.as_ptr(),
            &0,
            &std::ptr::null(),
        );
        if status != 0 {
            return match errstr(self.env.inner, status) {
                Ok(s) => Err(s),
                Err(e) => Err(e),
            };
        } else {
            Ok(())
        }
    }

    /// Solve the Problem, returning a `Solution` object with the
    /// result.
    pub fn solve_as(&mut self, pt: ProblemType) -> Result<Solution, String> {
        // TODO: support multiple solution types...

        if pt == ProblemType::Linear {
            let status = CPLEX_API.CPXchgprobtype(self.env.inner, self.inner, 0);

            if status != 0 {
                return Err(format!(
                    "Failed to convert to LP problem ({} ({}))",
                    errstr(self.env.inner, status).unwrap(),
                    status
                ));
            }
        }
        let status = match pt {
            ProblemType::MixedInteger => CPLEX_API.CPXmipopt(self.env.inner, self.inner),
            ProblemType::Linear => CPLEX_API.CPXlpopt(self.env.inner, self.inner),
        };
        if status != 0 {
            CPLEX_API.CPXwriteprob(
                self.env.inner,
                self.inner,
                CString::new("lpex1.lp").unwrap().as_ptr(),
                std::ptr::null(),
            );
            return Err(format!(
                "LP Optimization failed ({} ({}))",
                errstr(self.env.inner, status).unwrap(),
                status
            ));
        }

        let mut objval: f64 = 0.0;
        let status = CPLEX_API.CPXgetobjval(self.env.inner, self.inner, &mut objval);
        if status != 0 {
            CPLEX_API.CPXwriteprob(
                self.env.inner,
                self.inner,
                CString::new("lpex1.lp").unwrap().as_ptr(),
                std::ptr::null(),
            );
            return Err(format!(
                "Failed to retrieve objective value ({} ({}))",
                errstr(self.env.inner, status).unwrap(),
                status
            ));
        }

        let mut xs = vec![0f64; self.variables.len()];
        let status = CPLEX_API.CPXgetx(
            self.env.inner,
            self.inner,
            xs.as_mut_ptr(),
            0,
            self.variables.len() as CInt - 1,
        );
        if status != 0 {
            return Err(format!(
                "Failed to retrieve values for variables ({} ({}))",
                errstr(self.env.inner, status).unwrap(),
                status
            ));
        }

        let mut eps = EPINT;
        CPLEX_API.CPXgetdblparam(self.env.inner, EPINT_ID, &mut eps);

        return Ok(Solution {
            objective: objval,
            variables: xs
                .iter()
                .zip(self.variables.iter())
                .map(|(&x, v)| match v.ty {
                    VariableType::Binary => VariableValue::Binary(x <= 1.0 + eps && x >= 1.0 - eps),
                    VariableType::Continuous => VariableValue::Continuous(x),
                    VariableType::Integer => VariableValue::Integer(x as CInt),
                    VariableType::SemiContinuous => VariableValue::SemiContinuous(x),
                    VariableType::SemiInteger => VariableValue::SemiInteger(x as CInt),
                })
                .collect::<Vec<VariableValue>>(),
        });
    }

    /// Solve the problem as a Mixed Integer Program
    pub fn solve(&mut self) -> Result<Solution, String> {
        self.solve_as(ProblemType::MixedInteger)
    }
}

impl<'a> Drop for Problem<'a> {
    fn drop(&mut self) {
        assert!(CPLEX_API.CPXfreeprob(self.env.inner, &self.inner) == 0);
    }
}
