// rusanta-tensor/src/autograd/node.rs

//! Autograd computation graph nodes.
//!
//! Each operation performed on a Tensor creates a Node.
//!
//! Example:
//!
//! ```text
//!
//! a ----
//!       \
//!        Add Node ---- c
//!       /
//! b ----
//!
//! ```
//!
//! During backward pass:
//!
//! ```text
//! c.grad
//!   |
//!   v
//! Add Node backward()
//!   |
//!   +---- a.grad
//!   |
//!   +---- b.grad
//!
//! ```


use std::sync::{
    Arc,
    Mutex,
    Weak,
};


use crate::Tensor;





// =====================================================
// Operation Types
// =====================================================


/// Operation that produced a tensor.
#[derive(
    Debug,
    Clone,
)]
pub enum Op {


    /// Tensor was created manually.
    Leaf,



    /// Addition.
    Add,



    /// Subtraction.
    Sub,



    /// Multiplication.
    Mul,



    /// Matrix multiplication.
    MatMul,



    /// Division.
    Div,



    /// Power operation.
    Pow,



    /// Exponential.
    Exp,



    /// Natural logarithm.
    Log,



    /// Activation functions.
    ReLU,

    Sigmoid,

    Tanh,



    /// Reduction.
    Sum,

    Mean,



    /// Neural network layer.
    Linear,



    /// Loss functions.
    MSELoss,

    CrossEntropy,

}









// =====================================================
// Node Definition
// =====================================================


/// A node inside the autograd graph.
///
/// Stores:
///
/// - operation type
/// - input tensors
/// - gradient state
/// - parent relationships
///
/// Similar concept:
///
/// PyTorch:
/// `grad_fn`
pub struct Node {


    /// Operation creating this node.
    pub op:Op,



    /// Parent nodes.
///
/// These are tensors required to compute this node.
    parents:
        Vec<Weak<Mutex<Node>>>,




    /// Output tensor.
///
/// Used during backward propagation.
    pub output:
        Option<Tensor>,




    /// Whether node already received gradient.
    pub visited:
        bool,



}







impl Node {


    /// Create a new graph node.
    pub fn new(
        op:Op,
    )
        -> Self
    {

        Self {

            op,

            parents:
                Vec::new(),

            output:
                None,

            visited:
                false,

        }

    }







    /// Add parent dependency.
    pub fn add_parent(
        &mut self,
        parent:
            &Arc<Mutex<Node>>,
    )
    {

        self.parents
            .push(
                Arc::downgrade(parent)
            );

    }






    /// Get parents.
    pub fn parents(
        &self,
    )
        -> Vec<Arc<Mutex<Node>>>
    {

        self.parents
            .iter()
            .filter_map(
                |p|
                p.upgrade()
            )
            .collect()

    }






    /// Assign output tensor.
    pub fn set_output(
        &mut self,
        tensor:Tensor,
    )
    {

        self.output =
            Some(tensor);

    }





    /// Get operation type.
    pub fn operation(
        &self,
    )
        -> &Op
    {

        &self.op

    }



}

use std::sync::Arc;





// =====================================================
// Backward Function
// =====================================================



/// Backward computation function.
///
/// Every operation stores how gradients flow backwards.
///
/// Example:
///
/// Add:
///
/// ```text
/// z = x + y
///
/// dz/dx = 1
/// dz/dy = 1
/// ```
///
/// Mul:
///
/// ```text
/// z = x * y
///
/// dz/dx = y
/// dz/dy = x
/// ```
///
/// Future implementations can replace this with
/// optimized kernel dispatch.
pub type BackwardFn = Arc<
    dyn Fn(
        &Tensor
    ) -> Vec<Tensor>
    + Send
    + Sync
>;









// =====================================================
// Extended Node Implementation
// =====================================================


impl Node {



    /// Create operation node.
    ///
    /// Used internally by ops.
    pub fn operation_node(
        op:Op,
        parents:
            Vec<Arc<Mutex<Node>>>,
        backward:
            Option<BackwardFn>,
    )
        -> Arc<Mutex<Node>>
    {


        Arc::new(
            Mutex::new(
                Node {

                    op,

                    parents:
                        parents
                            .iter()
                            .map(
                                |p|
                                Arc::downgrade(p)
                            )
                            .collect(),


                    output:
                        None,


                    visited:
                        false,


                    backward,

                }
            )
        )

    }







    /// Create leaf node.
    ///
    /// Leaf tensors are tensors created directly
    /// by the user.
    ///
    /// Example:
    ///
    /// ```rust
    /// let x = Tensor::ones(...);
    /// ```
    ///
    /// x is a leaf.
    pub fn leaf()
        -> Arc<Mutex<Node>>
    {

        Arc::new(
            Mutex::new(
                Node {

                    op:
                        Op::Leaf,


                    parents:
                        Vec::new(),


                    output:
                        None,


                    visited:
                        false,


                    backward:
                        None,

                }
            )
        )

    }







    /// Execute backward function.
    pub fn apply_backward(
        &self,
        grad:&Tensor,
    )
        -> Vec<Tensor>
    {

        match &self.backward {


            Some(func)=>
                func(
                    grad
                ),



            None=>
                Vec::new(),

        }

    }







    /// Mark node as visited.
    pub fn mark_visited(
        &mut self,
    )
    {

        self.visited=true;

    }






    /// Reset traversal state.
    pub fn reset(
        &mut self,
    )
    {

        self.visited=false;

    }








    /// Check if node is leaf.
    pub fn is_leaf(
        &self,
    )
        -> bool
    {

        matches!(
            self.op,
            Op::Leaf
        )

    }





}







// =====================================================
// Node Debug
// =====================================================


impl std::fmt::Debug for Node {


    fn fmt(
        &self,
        f:&mut std::fmt::Formatter<'_>,
    )
        -> std::fmt::Result
    {


        f.debug_struct(
            "Node"
        )


        .field(
            "operation",
            &self.op
        )


        .field(
            "parents",
            &self.parents.len()
        )


        .field(
            "visited",
            &self.visited
        )


        .finish()

    }

}
