use std::{collections::HashMap, ffi::c_int};

use tensorflow::{
    DataType, FetchToken, Graph, Operation, Result, SavedModelBundle, SessionOptions, SignatureDef, Tensor, TensorInfo,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

#[derive(Debug)]
struct OpAndIndexAndType {
    op: Operation,
    index: c_int,
    d_type: DataType,
}

#[derive(Debug)]
struct SavedModel {
    bundle: SavedModelBundle,
    input_ops: HashMap<String, OpAndIndexAndType>,
    output_ops: HashMap<String, OpAndIndexAndType>,
}

enum TensorModelTypes {
    Float32(Tensor<f32>),
    String(Tensor<String>),
    Float64(Tensor<f64>),
    Uint64(Tensor<u64>),
}

impl SavedModel {
    fn load(model_path: &str, inputs: &Vec<&str>, outputs: &Vec<&str>) -> SavedModel {
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_path)
                .expect("could not load saved model");

        let serving_sig = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .expect("could not find serving_default");

        let tf_inputs: HashMap<String, OpAndIndexAndType> = inputs
            .into_iter()
            .map(|input| {
                (
                    input,
                    get_sig_name(input, serving_sig, SignatureDef::get_input),
                )
            })
            .map(|tuple| (tuple.0.to_string(), get_operation(&graph, tuple.1)))
            .collect();

        let tf_outputs: HashMap<String, OpAndIndexAndType> = outputs
            .into_iter()
            .map(|input| {
                (
                    input,
                    get_sig_name(input, serving_sig, SignatureDef::get_output),
                )
            })
            .map(|tuple| (tuple.0.to_string(), get_operation(&graph, tuple.1)))
            .collect();

        return SavedModel {
            bundle,
            input_ops: tf_inputs,
            output_ops: tf_outputs,
        };
    }

    fn predict(
        self: &Self,
        inputs: HashMap<String, TensorModelTypes>,
    ) -> Result<HashMap<String, Result<TensorModelTypes>>> {
        let session = &self.bundle.session;
        let mut sess_run = tensorflow::SessionRunArgs::new();

        inputs
            .iter()
            .for_each(|(key, value)| {
                match value {
                    TensorModelTypes::Float32(v) => {
                        sess_run.add_feed(&self.input_ops[key].op, self.input_ops[key].index, v);
                    },
                    TensorModelTypes::String(v) => {
                        sess_run.add_feed(&self.input_ops[key].op, self.input_ops[key].index, v);
                    }
                    TensorModelTypes::Float64(v) => {
                        sess_run.add_feed(&self.input_ops[key].op, self.input_ops[key].index, v)
                    }
                    TensorModelTypes::Uint64(v) => {
                        sess_run.add_feed(&self.input_ops[key].op, self.input_ops[key].index, v)
                    }
                }
            });

        let fetch_token_map: HashMap<String, FetchToken> = self
            .output_ops
            .iter()
            .map(|(key, value)| -> (String, FetchToken) {
                let fetch = sess_run.request_fetch(&value.op, value.index);
                (key.to_string(), fetch)
            })
            .collect();

        session.run(&mut sess_run)?;

        //here comes a loop over outputs and their mappings to model supported types
        let result: HashMap<String, Result<TensorModelTypes>> = fetch_token_map
            .into_iter()
            .map(|(key, fetch_token)| -> Result<(String, Result<TensorModelTypes>)> {
                let d_type: &DataType = &self.output_ops[&key].d_type;
                let value = match d_type {
                    DataType::Float => {
                        let fetch = sess_run.fetch(fetch_token)?;
                        Ok(TensorModelTypes::Float32(fetch))
                    }
                    DataType::Double => {
                        let fetch = sess_run.fetch(fetch_token)?;
                        Ok(TensorModelTypes::Float64(fetch))
                    }
                    DataType::String => {
                        let fetch = sess_run.fetch(fetch_token)?;
                        Ok(TensorModelTypes::String(fetch))
                    }
                    DataType::UInt64 => {
                        let fetch = sess_run.fetch(fetch_token)?;
                        Ok(TensorModelTypes::Uint64(fetch))
                    }
                    _ => panic!("not supported datatype"),
                };
                Ok((key, value))
            })
            .filter_map(|item| item.ok())
            .collect();
        return Ok(result);
    }
}

fn get_sig_name<'a, 'b>(
    arg: &'b str,
    sig_def: &'a tensorflow::SignatureDef,
    f: fn(&'a tensorflow::SignatureDef, &'b str) -> tensorflow::Result<&'a TensorInfo>,
) -> &'a TensorInfo {
    f(sig_def, arg).expect(format!("could not get info for {}", arg).as_str())
}

fn get_operation(graph: &Graph, field: &TensorInfo) -> OpAndIndexAndType {
    let operation = graph
        .operation_by_name_required(&field.name().name.as_str())
        .expect(&format!(
            "could not get operation with name: {}",
            field.name().name.as_str()
        ));

    OpAndIndexAndType {
        op: operation,
        index: field.name().index,
        d_type: field.dtype(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_saved_model() {
        let filename = "tf_model_market_model/dummymodel/";
        let inputs = vec![
            "mediation_name",
            "country",
            "model_type",
            "game_id",
            "session_depth",
            "max_bid",
            "original_rev_share",
        ];
        let outputs = vec![
            "optimized_rev_share",
            "log_str",
            "prob_win_for_optimized_bid",
            "prob_win_for_economical_bid",
            "optimized_bid",
            "economical_bid",
            "model_name",
            "model_version",
            "model_timestamp",
        ];

        let model = SavedModel::load(filename, &inputs, &outputs);
        assert_eq!(model.input_ops.len(), inputs.len());
        assert_eq!(model.output_ops.len(), outputs.len());

        println!("inputs {:?}", model.input_ops);
        println!("outputs {:?}", model.output_ops);

        // All inputs have index 0
        for inp in inputs {
            assert_eq!(model.input_ops[inp].index, 0);
        }

        assert_eq!(model.output_ops["model_timestamp"].index, 3);
        assert_eq!(model.output_ops["prob_win_for_economical_bid"].index, 7);
        assert_eq!(model.output_ops["optimized_rev_share"].index, 6);
        assert_eq!(model.output_ops["log_str"].index, 1);
        assert_eq!(model.output_ops["prob_win_for_optimized_bid"].index, 8);
        assert_eq!(model.output_ops["optimized_bid"].index, 5);
        assert_eq!(model.output_ops["economical_bid"].index, 0);
        assert_eq!(model.output_ops["model_version"].index, 4);
        assert_eq!(model.output_ops["model_name"].index, 2);
    }

    #[test]
    fn test_model_predict() {
        let filename = "tf_model_market_model/dummymodel/";
        let inputs = vec![
            "mediation_name",
            "country",
            "model_type",
            "game_id",
            "session_depth",
            "max_bid",
            "original_rev_share",
        ];
        let outputs = vec![
            "optimized_rev_share",
            "log_str",
            "prob_win_for_optimized_bid",
            "prob_win_for_economical_bid",
            "optimized_bid",
            "economical_bid",
            "model_name",
            "model_version",
            "model_timestamp",
        ];

        let model = SavedModel::load(filename, &inputs, &outputs);

        let results = model.predict(inputs);
    }
}
