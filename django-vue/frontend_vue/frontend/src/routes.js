import { createRouter, createWebHistory } from 'vue-router'
import Pods from './views/Pods'
import Resources from './views/Resources'
import Login from './views/Login'
import Logout from './views/Logout'
import Home from './views/Home'
import Schedule from './views/Schedule'
import PodView from './views/PodView'
import Activities from './views/Activities'
import Developers from './views/Developers'




const routes = [
    {
        path: '/pods',
        name: 'pods',
        component: Pods,
        meta: {
            requiresLogin: true
        }
    },
    {
        path: '/resources',
        name: 'resources',
        component: Resources,
        meta: {
            requiresLogin: true
        }
    },
    {
        path: '/log-in',
        name: 'log-in',
        component: Login,
    },
    {
        path: '/log-out',
        name: 'log-out',
        component: Logout,
    },
    {
        path: '/',
        name: 'home',
        component: Home,
        meta: {
            requiresLogin: true
        }
    },
    {
        path: '/schedule',
        name: 'schedule',
        component: Schedule,
        meta: {
            requiresLogin: true
        }
    },
    {
        path: '/pod-view',
        name: 'pod-view',
        component: PodView,
        meta: {
            requiresLogin: true
        }
    },
    {
        path: '/activities',
        name: 'activities',
        component: Activities,
        meta: {
            requiresLogin: true
        }
    },
    {
        path: '/developers',
        name: 'developers',
        component: Developers,
        meta: {
            requiresLogin: true
        }
    },
    
    
]
const router = createRouter({
    history: createWebHistory(process.env.BASE_URL),
    routes
})

export default router
