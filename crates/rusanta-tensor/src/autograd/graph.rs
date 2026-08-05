// rusanta-tensor/src/autograd/graph.rs

//! Computation graph management.
//!
//! The graph stores the relationship between tensor
//! operations and allows the backward engine to traverse
//! dependencies in reverse order.


use std::sync::{
    Arc,
    Mutex,
};


use crate::autograd::node::Node;





// =====================================================
// Graph Definition
// =====================================================



/// Dynamic computation graph.
///
/// Similar concepts:
///
/// - PyTorch Dynamic Graph
/// - TensorFlow GradientTape
pub struct Graph {


    /// Stored computation nodes.
    nodes:
        Vec<Arc<Mutex<Node>>>,


}





impl Graph {


    /// Create empty graph.
    pub fn new()
        -> Self
    {

        Self {

            nodes:
                Vec::new(),

        }

    }






    /// Register node inside graph.
    pub fn add(
        &mut self,
        node:
            Arc<Mutex<Node>>,
    )
    {

        self.nodes
            .push(node);

    }






    /// Number of nodes.
    pub fn len(
        &self,
    )
        -> usize
    {

        self.nodes.len()

    }





    /// Check empty graph.
    pub fn is_empty(
        &self,
    )
        -> bool
    {

        self.nodes.is_empty()

    }








    /// Clear graph.
    ///
    /// Used after backward when
    /// retaining graph is not requested.
    pub fn clear(
        &mut self,
    )
    {

        self.nodes.clear();

    }







    /// Get all nodes.
    pub fn nodes(
        &self,
    )
        -> &[Arc<Mutex<Node>>]
    {

        &self.nodes

    }






}









// =====================================================
// Graph Traversal
// =====================================================


impl Graph {


    /// Build reverse topological ordering.
    ///
    /// Example:
    ///
    /// ```
    /// a -> b -> c
    ///
    /// result:
    ///
    /// [c,b,a]
    ///
    /// ```
    pub fn backward_order(
        &self,
        root:
            Arc<Mutex<Node>>,
    )
        -> Vec<Arc<Mutex<Node>>>
    {


        let mut visited =
            Vec::new();


        let mut order =
            Vec::new();



        Self::visit(
            root,
            &mut visited,
            &mut order,
        );



        order.reverse();


        order

    }







    fn visit(
        node:
            Arc<Mutex<Node>>,
        visited:
            &mut Vec<usize>,
        order:
            &mut Vec<Arc<Mutex<Node>>>,
    )
    {


        let id =
            Arc::as_ptr(&node)
                as usize;



        if visited.contains(&id) {

            return;

        }


        visited.push(id);





        let parents = {

            let guard =
                node.lock()
                    .unwrap();


            guard.parents()

        };





        for parent in parents {


            Self::visit(
                parent,
                visited,
                order,
            );


        }



        order.push(node);

    }







}









// =====================================================
// Global Graph Helper
// =====================================================



impl Default for Graph {


    fn default()
        -> Self
    {
        Self::new()
    }

}
