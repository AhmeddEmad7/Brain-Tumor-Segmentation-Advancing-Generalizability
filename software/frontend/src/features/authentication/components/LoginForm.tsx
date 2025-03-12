import { FormEvent, useRef, useState } from 'react';
import { Button, Theme } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import LoginInputField from '@features/authentication/components/LoginInputField.tsx';
import { Person as UsernameIcon, Lock as PasswordIcon } from '@mui/icons-material';

const LoginForm = ({ theme }: { theme: Theme }) => {
    const loginFormRef = useRef<HTMLFormElement>(null);
    const navigate = useNavigate();
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(null);

        if (loginFormRef.current) {
            const formData = new FormData(loginFormRef.current);
            const username = formData.get('username');
            const password = formData.get('password');

            // Check for the correct credentials
            if (username === 'hazem11' && password === 'mina123') {
                console.log('Login successful');
                navigate('/'); // Redirect to viewer page
            } else {
                setError('Invalid username or password');
            }
        }
    };

    return (
        <form ref={loginFormRef} onSubmit={handleSubmit}>
            <LoginInputField
                theme={theme}
                name="username"
                autoComplete="username"
                placeholder="Username"
                Icon={UsernameIcon}
            />

            <LoginInputField
                theme={theme}
                name="password"
                autoComplete="current-password"
                placeholder="Password"
                Icon={PasswordIcon}
                type="password"
            />

            {error && <p style={{ color: 'red', textAlign: 'center' }}>{error}</p>}

            <Button
                type="submit"
                fullWidth
                size="large"
                color="primary"
                className="h-12 text-2"
                style={{ backgroundColor: theme.palette.secondary.main }}
            >
                Login
            </Button>
        </form>
    );
};

export default LoginForm;
