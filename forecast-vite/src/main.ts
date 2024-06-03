import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import './index.css'
import PrimeVue from 'primevue/config';

const app = createApp(App);
app.use(PrimeVue);
app.mount('#app');
// createApp(App).mount('#app')
