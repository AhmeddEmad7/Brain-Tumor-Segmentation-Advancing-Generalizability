import { useState } from 'react';
import { InputAdornment, OutlinedInput, SvgIcon, IconButton, Theme } from '@mui/material';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import { ReactNode } from 'react';

interface LoginInputFieldProps {
    name: string;
    autoComplete: string;
    placeholder: string;
    Icon: (props: any) => ReactNode;
    theme: Theme;
    type?: string;
}

const LoginInputField = ({ name, autoComplete, placeholder, Icon, theme, type = 'text' }: LoginInputFieldProps) => {
    const [showPassword, setShowPassword] = useState(false);

    return (
        <OutlinedInput
            defaultValue=""
            name={name}
            fullWidth
            type={type === 'password' && !showPassword ? 'password' : 'text'} // Toggle password visibility
            autoComplete={autoComplete}
            placeholder={placeholder}
            className={'my-2'}
            startAdornment={
                <InputAdornment position="start">
                    <SvgIcon color="inherit" fontSize="large">
                        <Icon />
                    </SvgIcon>
                </InputAdornment>
            }
            endAdornment={
                type === 'password' && (
                    <InputAdornment position="end">
                        <IconButton
                            onClick={() => setShowPassword((prev) => !prev)}
                            edge="end"
                            aria-label="toggle password visibility"
                        >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                    </InputAdornment>
                )
            }
            sx={{
                backgroundColor: theme.palette.primary.dark,
                fontSize: 'medium',
                '& .MuiOutlinedInput-notchedOutline': {
                    border: 'none' // Remove the outline
                }
            }}
        />
    );
};

export default LoginInputField;
