// rusanta-tensor/src/autograd/backward.rs

//! Automatic differentiation backward engine.
//!
//! Responsible for propagating gradients through the
//! computation graph.

use std::collections::HashMap;
use std::sync::{
    Arc,
    Mutex,
};


use crate::{
    Tensor,
    DType,
    Device,
    Result,
    TensorError,
};


use crate::autograd::{
    Graph,
    Node,
};







// =====================================================
// Public Backward API
// =====================================================



/// Execute reverse-mode automatic differentiation.
///
/// Example:
///
/// ```text
///
/// y = x*x
///
/// y.backward()
///
/// gives:
///
/// dy/dx = 2x
///
/// ```
pub fn backward(
    tensor:&Tensor,
)
    -> Result<()>
{

    if !tensor.requires_grad() {

        return Err(
            TensorError::AutogradError(
                "Tensor does not require gradients"
                    .into()
            )
        );

    }



    let node =
        match tensor.node() {


            Some(node)=>
                node,



            None=>{

                return Err(
                    TensorError::AutogradError(
                        "Tensor has no computation graph"
                            .into()
                    )
                );

            }

        };




    let graph =
        Graph::new();




    let order =
        graph.backward_order(
            node
        );




    let initial_grad =
        Tensor::ones(
            tensor.shape().dims(),
            tensor.dtype(),
            Device::CPU,
        );



    execute(
        order,
        initial_grad,
    );



    Ok(())

}









// =====================================================
// Backward Execution
// =====================================================



fn execute(
    nodes:
        Vec<Arc<Mutex<Node>>>,

    initial_grad:
        Tensor,
)
{

    let mut gradients:
        HashMap<
            usize,
            Tensor
        >
        =
        HashMap::new();




    //
    // Root output gradient.
    //
    if let Some(last)=nodes.last()
    {

        let id =
            Arc::as_ptr(last)
                as usize;


        gradients.insert(
            id,
            initial_grad,
        );

    }






    for node in nodes.iter().rev()
    {

        let id =
            Arc::as_ptr(node)
                as usize;



        let grad =
            match gradients.remove(&id)
            {

                Some(g)=>
                    g,


                None=>
                    continue,

            };




        let parents =
        {

            let guard =
                node.lock()
                    .unwrap();


            guard.parents()

        };



        if parents.is_empty()
        {
            continue;
        }





        let backward_result =
        {

            let guard =
                node.lock()
                    .unwrap();


            guard.apply_backward(
                &grad
            )

        };




        for (parent, parent_grad)
            in parents
                .into_iter()
                .zip(
                    backward_result.into_iter()
                )
        {


            let parent_id =
                Arc::as_ptr(&parent)
                    as usize;



            gradients
                .entry(parent_id)
                .and_modify(
                    |existing|
                    {
                        accumulate(
                            existing,
                            &parent_grad
                        );
                    }
                )
                .or_insert(
                    parent_grad
                );


        }

    }

}









// =====================================================
// Gradient Accumulation
// =====================================================


fn accumulate(
    a:&mut Tensor,
    b:&Tensor,
)
{

    match (
        a.storage_mut(),
        b.storage(),
    )
    {


        (
            crate::tensor::Storage::F32(x),
            crate::tensor::Storage::F32(y)
        )=>{

            for i in 0..x.len()
            {

                x[i]+=y[i];

            }

        }




        (
            crate::tensor::Storage::F64(x),
            crate::tensor::Storage::F64(y)
        )=>{

            for i in 0..x.len()
            {

                x[i]+=y[i];

            }

        }




        _=>{}

    }

}


use crate::autograd::node::Op;







// =====================================================
// Backward Configuration
// =====================================================



/// Backward execution options.
#[derive(
    Debug,
    Clone,
    Copy
)]
pub struct BackwardConfig {


    /// Keep computation graph after backward.
    ///
    /// Useful for:
    ///
    /// - higher order gradients
    /// - repeated backward calls
    pub retain_graph:bool,

}



impl Default for BackwardConfig {


    fn default()
        -> Self
    {

        Self {

            retain_graph:false,

        }

    }

}








/// Execute backward with options.
pub fn backward_with_config(
    tensor:&Tensor,
    config:BackwardConfig,
)
    -> Result<()>
{


    if !tensor.requires_grad()
    {

        return Err(
            TensorError::AutogradError(
                "Tensor does not require gradients"
                    .into()
            )
        );

    }



    let node =
        tensor.node()
            .ok_or_else(
                || {

                    TensorError::AutogradError(
                        "Missing computation graph"
                            .into()
                    )

                }
            )?;





    let graph =
        Graph::new();



    let order =
        graph.backward_order(
            node
        );





    let gradient =
        Tensor::ones(
            tensor.shape().dims(),
            tensor.dtype(),
            tensor.device(),
        );




    execute_with_leaf(
        order,
        gradient,
    );





    if !config.retain_graph
    {

        //
        // Future:
        //
        // Release graph references
        // after backward.
        //

    }



    Ok(())

}









// =====================================================
// Extended Backward Execution
// =====================================================



fn execute_with_leaf(
    nodes:
        Vec<Arc<Mutex<Node>>>,

    initial:
        Tensor,
)
{


    let mut gradients:
        HashMap<
            usize,
            Tensor
        >
        =
        HashMap::new();





    if let Some(root)=nodes.last()
    {


        let id =
            Arc::as_ptr(root)
                as usize;


        gradients.insert(
            id,
            initial,
        );


    }








    for node in nodes.iter().rev()
    {


        let id =
            Arc::as_ptr(node)
                as usize;



        let grad =
            match gradients.remove(&id)
            {


                Some(value)=>
                    value,


                None=>
                    continue,


            };






        let (
            parents,
            operation,
            is_leaf,
        ) =
        {

            let guard =
                node.lock()
                    .unwrap();



            (
                guard.parents(),
                guard.operation().clone(),
                guard.is_leaf(),
            )

        };








        //
        // Leaf reached.
        //
        // In future this will call:
        //
        // Tensor::accumulate_grad()
        //
        //
        if is_leaf
        {

            continue;

        }







        let produced_grads =
        {


            let guard =
                node.lock()
                    .unwrap();



            guard.apply_backward(
                &grad
            )

        };







        for (
            parent,
            parent_grad
        )
        in parents
            .into_iter()
            .zip(
                produced_grads
            )
        {


            let parent_id =
                Arc::as_ptr(&parent)
                    as usize;



            gradients
                .entry(parent_id)
                .and_modify(
                    |existing|
                    {

                        accumulate(
                            existing,
                            &parent_grad
                        );

                    }
                )
                .or_insert(
                    parent_grad
                );



        }






        //
        // Future operation-specific
        // optimization hooks.
        //
        match operation
        {

            Op::MatMul => {

                // optimized matmul backward
                // will be added later

            }


            Op::Linear => {

                // neural network layer backward

            }


            _=>{}

        }


    }


}
