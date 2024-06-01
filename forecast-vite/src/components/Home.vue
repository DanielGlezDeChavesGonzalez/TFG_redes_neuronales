<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import Chart from 'chart.js/auto';

const file = ref<File | null>(null);

const data = ref<number[]>([]);
let first_value = '';
let second_value = '';

const uploadFileCsv = (event: Event) => {
    const target = event.target as HTMLInputElement;
    const files = target.files;

    if (files) {
        file.value = files[0];
    }

    if (file.value) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target?.result as string;

            const values = content.split('\n').map((value) => {
                // first_value = value.split(';')[0];
                second_value = value.split(';')[1];

                return parseFloat(value.split(';')[1]);
            });

            data.value = values;

            // console.log(data.value);
        };

        reader.readAsText(file.value);
    }
};

const submitFile = () => {
    if (file.value) {
        const formData = new FormData();
        formData.append('file', file.value);

        axios.post('http://localhost:5000/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        }).then((response) => {
            console.log(response);
        }).catch((error) => {
            console.error(error);
        });
    }
};
</script>

<template>
    <div class="container">
        <h1 class="title">Upload CSV File for predictions</h1>
        <div v-if="file == null" class="submiting-form">
            <p>Upload a CSV file to predict the future values of a time series</p>

            <p>File must have only one column with the values</p>

            <p>File must have at least 32 values</p>

            <p>It will predict next 10 steps</p>

            <input type="file" class="input-file border" @change="uploadFileCsv" />

            <button class="button" @click="submitFile">Submit</button>
        </div>

        <p v-if="file">File name: {{ file.name }}</p>
        <p v-if="file">Data info: {{ data }}</p>

        <div v-if="file" class="graph-predictions">
            <h2>Predictions</h2>

            <p>Graph with the predictions</p>

            <p>Table with the predictions</p>

            <p>Download the predictions</p>

            <!-- // print a chart with the predictions -->
            <canvas id="chart"></canvas>

            <button class="button" @click="file = null">Upload another file</button>

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
    margin: 10px 0;
}

.input-file {
    margin: 10px 0;
}
</style>