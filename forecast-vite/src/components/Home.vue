<script setup lang="ts">
import { ref, onMounted } from 'vue';
import axios from 'axios';
import Chart from 'primevue/chart';
// import FileUpload from 'primevue/fileupload';

const file = ref<File | null>(null);
const sumbmited = ref<boolean>(false);

const task_id = ref<string>('');
const success = ref<boolean>(false);

const predictions = ref<number[]>([]);

const data = ref<number[]>([]);

const checkFile = (file: File) => {
    if (file.type !== 'text/csv') {
        alert('File must be a CSV file');
        return false;
    }
    return true;
};

const uploadFileCsv = (event: Event) => {
    const target = event.target as HTMLInputElement;
    const files = target.files;
    if (files) {
        file.value = files[0];
    }
    if (file.value) {
        if (!checkFile(file.value)) {
            file.value = null;
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target?.result as string;
            const values = content.split('\n').map((value) => {
                return parseFloat(value.split(';')[1]);
            });
            data.value = values;
            // console.log(data.value);
        };
        reader.readAsText(file.value);
    }
};

const gotResults = ref<boolean>(false);

const submitFile = () => {
    if (file.value) {

        const payload = {
            data: data.value,
        };

        // console.log(payload);

        axios.post('http://localhost:5000/predict', payload)
            .then((response) => {
        
                task_id.value = response.data.task_id;
                success.value = response.data.success;
                // console.log("task_id: ", task_id.value);
                // console.log("success: ", success.value);
                sumbmited.value = true;
                if (success.value) {
                    console.log('Success');
                    setTimeout(() => {
                        getResults();
                    }, 3000);
                } else {
                    console.log('Error');
                }
                console.log(response);
            })
            .catch((error) => {
                console.error(error);
            });
    }
};

const getResults = () => {
    axios.get(`http://localhost:5000/predict/results/${task_id.value}`).then((response) => {
        // Predictions: [ [ 0.9034325480461121, 0.9265543222427368, 0.9013111591339111, 0.9097467064857483, 0.8530929088592529 ] ]
        // console.log(response.data);
        predictions.value = response.data.predictions;
        // console.log(predictions.value);
        // console.log("predictions: ", predictions.value);
        chartData.value = setChartData();
        console.log(response);
    }).catch((error) => {
        console.error(error);
    });
    gotResults.value = true;
};


// ECHARLE UNA PENSADA A ESTO PARA QUE NO SE QUEDE EN UN LOOP INFINITO DE PETICIONES AL SERVIDOR
// watch(task_id, () => {
//     if (task_id.value) {
//         setInterval(() => {
//             getResults();
//         }, 5000);
//     }
// });

const chartData = ref();
const chartOptions = ref();

onMounted(() => {
    chartOptions.value = setChartOptions();
});

const setChartData = () => {
    return {
        labels: [...data.value.map((_, index) => index), ...predictions.value.map((_, index) => index + data.value.length)],
        datasets: [
            {
                label: 'Original Data',
                data: data.value,
                fill: false,
                borderColor: 'blue',
                borderWidth: 0.8,
            },
            {
                label: 'Predictions',
                data: [...data.value, ...predictions.value],
                fill: false,
                borderColor: 'red',
                borderWidth: 0.8,
            },
        ],
    };
};

const setChartOptions = () => {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Time',
                },
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: 'Value',
                },
            },
        },
    };
};

</script>

<template>
    <div class="container">
        <h1 class="flex text-3xl p-2 w-full justify-center border-b-4 border-black">Upload CSV File for predictions</h1>

        <div v-if="file">
        </div>


        <div v-if="file == null || !sumbmited" class="submiting-form mt-2">
            <input type="file" class="input-file" @change="uploadFileCsv" />
            <!-- <FileUpload mode="basic" @change="uploadFileCsv" @remove="file = null" accept=".csv"
                :maxFileSize="1000000" class="input-file">
                <template #empty>
                    <p>Drag and drop files to here to upload.</p>
                </template>
</FileUpload> -->
            <p>Upload a CSV file to predict the future values of a time series</p>

            <p>File must have only 2 columns with the first one being the index and the second one the value</p>

            <p>File must have at least 32 values</p>

            <p>It will predict next 10 steps</p>


            <p v-if="file" class="flex w-full justify-center mt-2 border-t-2 border-black">File name: {{ file.name }}
            </p>
            <p v-if="file" class="flex w-full justify-center border-b-2 border-black">Data Given: {{ data }}</p>
            <button class="button mt-4" @click="submitFile">Submit</button>
        </div>



        <div v-if="file && task_id" class="submiting-form-2 mt-2">
            <h2>Predictions</h2>

            <p>Task ID: {{ task_id }}</p>

            <p v-if="success">Success request</p>
            <p v-if="!success">Error</p>

            <div v-if="!gotResults">
                <p>Waiting for the predictions</p>

            </div>
            <div v-if="gotResults">
                <p>Got the predictions</p>
                <p>Predictions: {{ predictions }}</p>
                <p>Graph with the predictions</p>

                <Chart type="line" :data="chartData" :options="chartOptions" class="h-[300px] w-full" />
            </div>



            <!-- A simple line chart with the predictions the original data in blue and the predictions concatenated in red will be shown here the size will be 400px height and 100% width -->


            <button class="button"
                @click="file = null; sumbmited = false; task_id = ''; success = false; predictions = []">Reset</button>

        </div>
    </div>
</template>

<style scoped>
.container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px 20px;
    border: solid 1px black;
    border-radius: 10px;
    margin: auto;
    width: 1000px;
    height: auto;
}

.title {
    font-size: 2rem;
    margin: 10px 0;
    border-bottom: solid 1px black;
}

.submiting-form {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.submiting-form-2 {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 900px;
}

.input-file {
    margin: 10px 0;
}
</style>