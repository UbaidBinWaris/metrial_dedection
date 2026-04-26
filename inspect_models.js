const modelService = require('./src/services/modelService.js');

async function main() {
    try {
        await modelService.init();
        const models = modelService._models;
        
        models.forEach(m => {
            console.log(`Model: ${m.name}`);
            const input = m.interpreter.inputs[0];
            const output = m.interpreter.outputs[0];

            console.log('Input:');
            try { console.log('  type:', input.type); } catch(e) {}
            try { console.log('  dims:', JSON.stringify(input.dims)); } catch(e) {}
            try { console.log('  byteSize:', input.byteSize); } catch(e) {}

            console.log('Output:');
            try { console.log('  type:', output.type); } catch(e) {}
            try { console.log('  dims:', JSON.stringify(output.dims)); } catch(e) {}
            try { console.log('  byteSize:', output.byteSize); } catch(e) {}
            console.log('---');
        });
    } catch (err) {
        console.error(err);
    }
}

main();
