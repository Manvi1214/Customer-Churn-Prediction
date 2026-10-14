// JavaScript utility file for RetentionAI platform

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navigation scroll effect
    const navbar = document.getElementById('navbar');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            navbar.style.borderBottom = '1px solid rgba(99, 102, 241, 0.2)';
            navbar.style.backgroundColor = 'rgba(11, 15, 25, 0.9)';
        } else {
            navbar.style.borderBottom = '1px solid var(--border-color)';
            navbar.style.backgroundColor = 'rgba(11, 15, 25, 0.8)';
        }
    });

    // 2. Client-side form validation feedback
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', (e) => {
            const ageInput = document.getElementById('Age');
            const supportCallsInput = document.getElementById('Support Calls');
            const lastInteractionInput = document.getElementById('Last Interaction');
            
            if (ageInput && (ageInput.value < 18 || ageInput.value > 100)) {
                alert('Age must be between 18 and 100.');
                e.preventDefault();
                return;
            }
            
            if (supportCallsInput && (supportCallsInput.value < 0 || supportCallsInput.value > 20)) {
                alert('Support Calls must be between 0 and 20.');
                e.preventDefault();
                return;
            }
            
            if (lastInteractionInput && (lastInteractionInput.value < 0 || lastInteractionInput.value > 30)) {
                alert('Last Interaction must be between 0 and 30 days.');
                e.preventDefault();
                return;
            }
        });
    }
});
